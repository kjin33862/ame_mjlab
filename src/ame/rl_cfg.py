"""PPO runner configs for the AME and DELTA tasks.

``ame_ppo_runner_cfg`` is the reference (``agents/ame_rsl_rl_ppo_cfg.py``):
entropy 0.008, lr 1e-3 adaptive, γ 0.99, λ 0.95, clip 0.2, 5 epochs × 4
minibatches, 24 steps/env, 10k iterations, hidden dims (512, 256, 128),
no observation normalization.  The registered tasks train with
``ame_symmetric_ppo_runner_cfg`` / ``delta_ppo_runner_cfg``, which add the
AME tuning guide's symmetry augmentation and shared encoder.

The encoder rides in through :class:`AmeModelCfg` — its extra fields flow
into ``AmeModel.__init__`` as kwargs via the runner's config dict.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from ame.scan import MAP_SHAPE

NUM_MINI_BATCHES = 16
"""Minibatches per PPO epoch for the symmetry-augmented tasks.

Symmetry doubles each minibatch, and the CNN activations on it are the
largest torch allocation.  With the default 1024 envs, 16 minibatches gives
an augmented minibatch of 3072 samples and a ~9.6 GB training process (RTX
5080, measured); 8 minibatches needs ~2.7 GB more at the same speed.  On a
card with room to spare, raise ``scene.num_envs`` first and scale this with
it (2048 envs / 32 keeps the same minibatch).
"""


@dataclass
class AmeModelCfg(RslRlModelCfg):
  """Model cfg + AME encoder knobs (delivered to ``AmeModel`` as kwargs)."""

  map_group: str = "height_map"
  map_shape: tuple[int, int, int] = MAP_SHAPE
  mha_dim: int = 64
  num_heads: int = 16
  cnn_downsample: bool = True
  attach_global: bool = False
  """ame1 checkpoint config; ``True`` reproduces ame2 (global max-pool ctx)."""
  share_encoder: bool = False
  """Share map_cnn/mha between actor and critic (AME guide recommendation)."""


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
  """PPO cfg extended with rsl-rl's symmetry hook.

  ``RslRlPpoAlgorithmCfg`` declares no ``symmetry_cfg`` field, so setting the
  attribute on an instance is silently dropped by ``asdict()`` and never
  reaches the algorithm.  Declaring it here is what actually turns symmetry
  augmentation on.
  """

  symmetry_cfg: dict | None = None


@dataclass
class DeltaModelCfg(RslRlModelCfg):
  """Model cfg + DELTA encoder knobs (delivered to ``DeltaModel`` as kwargs).

  Defaults follow the paper's Table I (L=3, D=64, N_h=4, K=8, scout 3x3,
  final 9x9 with 3x3 center, gamma=0.25, FFN 256).  The three ablation
  switches mirror its Table VI and default to the full model.
  """

  map_group: str = "height_map"
  map_shape: tuple[int, int, int] = MAP_SHAPE
  embed_dim: int = 64
  num_layers: int = 3
  num_heads: int = 4
  """DELTA's own setting.  The AME guide's "doesn't work with fewer heads" was
  measured on AME's DENSE attention (one softmax over all map tokens); DELTA
  attends per-head over its own 8 sampled points — a different regime, and 16
  heads measured 3x update compute + 3.7x activation memory here.  Stay honest
  to the paper first; revisit heads only if sampling coverage proves limiting."""
  num_points: int = 8
  scout_patch: int = 3
  final_patch: int = 9
  final_center: int = 3
  offset_scale: float = 0.25
  ffn_hidden: int = 256
  enc_hidden: int = 64
  learned_offsets: bool = True
  scout_refine: bool = True
  adaptive_patch: bool = True
  share_encoder: bool = True
  """Guide: sharing the encoder between actor and critic speeds convergence,
  because the critic loss carries the better supervision signal."""


def delta_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The AME recipe with the DELTA encoder swapped in — nothing else moves."""
  cfg = ame_ppo_runner_cfg()
  cfg.actor = DeltaModelCfg(
    class_name="ame.delta:DeltaModel",
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  cfg.critic = DeltaModelCfg(
    class_name="ame.delta:DeltaModel",
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
  )
  cfg.experiment_name = "delta_g1"
  # Symmetry augmentation (guide: "It helps improve the motion style a lot").
  # It doubles every minibatch; the minibatch count is chosen by memory (see
  # ``NUM_MINI_BATCHES``) — update time barely depends on it.
  cfg.algorithm.num_mini_batches = NUM_MINI_BATCHES
  cfg.algorithm = PpoWithSymmetryCfg(
    **dataclasses.asdict(cfg.algorithm),
    symmetry_cfg={
      "data_augmentation_func": "ame.symmetry:mirror_obs_actions",
      "use_data_augmentation": True,
      "use_mirror_loss": False,
    },
  )
  # At desired_kl 0.01 the adaptive-KL controller pinned DELTA at the 1e-5
  # lr floor (while AME rode at ~1e-4); tracking plateaued and the curriculum
  # never promoted.  The attention + deformable-sampling policy is more
  # KL-sensitive per parameter step; widen the target so the controller can
  # settle at a working rate instead of the floor.
  cfg.algorithm.desired_kl = 0.02
  return cfg


def ame_symmetric_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The dense AME encoder under the full training recipe.

  Everything except the encoder matches :func:`delta_ppo_runner_cfg`, so the
  two tasks are a clean dense-vs-deformable A/B: symmetry augmentation,
  desired_kl 0.02, ``NUM_MINI_BATCHES`` minibatches, and the actor/critic
  encoder shared per the AME tuning guide.  Rewards, map noise and the guard
  terminations come from the env side (:func:`ame.env_cfg.ame_rough_env_cfg`).
  """
  cfg = ame_ppo_runner_cfg()
  for side in (cfg.actor, cfg.critic):
    assert isinstance(side, AmeModelCfg)
    side.share_encoder = True
  cfg.algorithm.desired_kl = 0.02
  cfg.algorithm.num_mini_batches = NUM_MINI_BATCHES
  cfg.algorithm = PpoWithSymmetryCfg(
    **dataclasses.asdict(cfg.algorithm),
    symmetry_cfg={
      "data_augmentation_func": "ame.symmetry:mirror_obs_actions",
      "use_data_augmentation": True,
      "use_mirror_loss": False,
    },
  )
  return cfg


def ame_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=AmeModelCfg(
      class_name="ame.models:AmeModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=AmeModelCfg(
      class_name="ame.models:AmeModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # Both networks attend the same map group next to their proprio group.
    obs_groups={
      "actor": ("actor", "height_map"),
      "critic": ("critic", "height_map"),
    },
    experiment_name="ame_g1",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=10_000,
    wandb_project="ame_mjlab",
  )


__all__ = [
  "NUM_MINI_BATCHES",
  "AmeModelCfg",
  "DeltaModelCfg",
  "PpoWithSymmetryCfg",
  "ame_ppo_runner_cfg",
  "ame_symmetric_ppo_runner_cfg",
  "delta_ppo_runner_cfg",
]
