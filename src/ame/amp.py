"""Adversarial Motion Priors (AMP) on top of :class:`ame.double_critic.DoubleCriticPPO`.

Reimplemented from AMP_mjlab (github.com/ccrpRepo/AMP_mjlab @ 6c7a294:
``rsl_rl/algorithms/amp_ppo.py``, ``rsl_rl/modules/discriminator.py``,
``rsl_rl/storage/replay_buffer.py``, ``rsl_rl/utils/motion_loader.py``,
``rsl_rl/runners/amp_on_policy_runner.py``, ``src/tasks/amp_loco``), itself
derived from RSL-RL / Legged Lab / TienKung-Lab.  That repository carries no
top-level LICENSE (its ``rsl_rl`` file headers say BSD-3-Clause), so no file is
copied: the logic is rewritten here for rsl_rl 5.4 and our double critic, and the
motion data is read from outside this repository (default
``~/ref/amp_mjlab_npz/amp``, override with ``AME_AMP_MOTION_DIR``).

What is the same as the reference (its defaults, ``config/g1/rl_cfg.py``):

* AMP observation (one frame, 195 dims): for 13 bodies (``AMP_BODY_NAMES``), the
  position and the first two rotation-matrix columns relative to ``torso_link``,
  then each body's linear and angular velocity in its own frame, term-major
  ``[pos 13x3 | ori 13x6 | lin_vel 13x3 | ang_vel 13x3]``.  A transition is two
  consecutive policy-step frames (discriminator input 390).
* Expert data: the reference's ``csv_to_npz.py`` output (120 fps -> 50 Hz = our
  policy dt 0.02 s).  Each expert minibatch is drawn from one clip, clips taken
  in sorted order cycling with the minibatch index, frames uniform, next frame =
  min(i + 1, T - 1).
* Discriminator: MLP 1024-512-256 ReLU + linear head; least-squares loss
  0.5 * (MSE(D(expert), 1) + MSE(D(policy), -1)); gradient penalty
  10 * |dD/dx|^2 on the *unnormalised* expert pairs (as the reference computes
  it); trained in the PPO optimizer (Adam, same adaptive learning rate) with
  weight decay 1e-3 on the trunk and 1e-1 on the head, one step per PPO
  minibatch, gradient clipping only on the policy.
* Input normaliser: running mean / var (count init 1e-4), (x - mean) /
  sqrt(var + 1e-4) clipped to +-10, updated after each step with the policy and
  expert *states* of that minibatch.
* Policy replay buffer: 100000 transitions, uniform sampling with replacement.
* Style reward: r_s = 0.1 * max(0, 1 - 0.25 * (D - 1)^2); at a terminal step
  the "next" frame is the pre-step frame (mjlab resets before observing).
* Mix: the reference returns (1 - 0.75) * r_s + 0.75 * r_task, i.e.
  ``task_reward_lerp`` 0.75 weights the TASK reward.  Here this applies to
  group 1 (the main critic): r1 <- 0.25 * r_s + 0.75 * r1.  Group 2 is left
  unscaled: its advantage is normalised by its own std, so a 0.75 factor there
  would change nothing but the value-loss scale.

What differs: the AMP observation is computed here from the robot's body state
(the same mjlab ``body_link_*`` quantities the reference's observation terms
read) instead of an observation group, so the env config is untouched; the
reference's ``min_normalized_std`` std floor and its PPO settings are not part
of AMP and are not adopted.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from ame.double_critic import DoubleCriticPPO

AMP_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
AMP_ANCHOR_NAME = "torso_link"
AMP_CLIPS = (
  # walk (8)
  "arc_walk_left_loop_001__A029",
  "walk_arc_cw_loop_002__A046",
  "walk_backward_loop_001__A022",
  "walk_backward_loop_001__A024",
  "walk_forward_loop_002__A022",
  "walk_forward_loop_002__A024",
  "walk_sideway_left_loop_002__A021",
  "walk_sideway_right_loop_001__A022",
  # jog (6)
  "arc_jog_left_loop_002__A029",
  "jog_arc_cw_loop_004__A045",
  "jog_backward_loop_002__A022",
  "jog_backward_loop_003__A023",
  "jog_forward_loop_003__A021",
  "jog_forward_loop_003__A022",
  # idle turn / step rotate (3)
  "idle_turn_270_001__A048",
  "idle_turn_360_001__A047",
  "step_rotate_idle_000_002__A026",
)
"""The 17 clips of AMP_mjlab ``motion_data_csv/amp`` (no fall / get-up clip)."""
AMP_MOTION_DIR = "~/ref/amp_mjlab_npz/amp"
AMP_MOTION_DIR_ENV = "AME_AMP_MOTION_DIR"
AMP_REWARD_COEF = 0.1
AMP_TASK_REWARD_LERP = 0.75
AMP_DISCR_HIDDEN_DIMS = (1024, 512, 256)
AMP_REPLAY_BUFFER_SIZE = 100_000
AMP_GRAD_PEN_LAMBDA = 10.0
AMP_TRUNK_WEIGHT_DECAY = 1.0e-3
AMP_HEAD_WEIGHT_DECAY = 1.0e-1


# ------------------------------------------------------------------ observation
def amp_observation(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
  """``[N, K, 3|4]`` body states and ``[N, 3|4]`` anchor -> ``[N, 15 K]``."""
  n, k = body_pos_w.shape[:2]
  pos_b, quat_b = subtract_frame_transforms(
    anchor_pos_w[:, None].expand(n, k, 3),
    anchor_quat_w[:, None].expand(n, k, 4),
    body_pos_w,
    body_quat_w,
  )
  ori_b = matrix_from_quat(quat_b)[..., :2].reshape(n, -1)
  lin_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_lin_vel_w.reshape(-1, 3))
  ang_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_ang_vel_w.reshape(-1, 3))
  return torch.cat(
    [pos_b.reshape(n, -1), ori_b, lin_b.reshape(n, -1), ang_b.reshape(n, -1)], dim=-1
  )


def resolve_motion_dir(motion_dir: str) -> Path:
  return Path(os.path.expanduser(os.environ.get(AMP_MOTION_DIR_ENV, motion_dir)))


class MotionData:
  """Expert AMP frames, one ``[T, D]`` tensor per clip (sorted by name)."""

  def __init__(
    self,
    motion_dir: str | Path,
    clips: Sequence[str],
    all_body_names: Sequence[str],
    body_names: Sequence[str] = AMP_BODY_NAMES,
    anchor_name: str = AMP_ANCHOR_NAME,
    device: str = "cpu",
  ) -> None:
    motion_dir = Path(motion_dir)
    names = list(all_body_names)
    body_ids = [names.index(n) for n in body_names]
    anchor = names.index(anchor_name)
    missing = [c for c in clips if not (motion_dir / f"{c}.npz").is_file()]
    if missing:
      raise FileNotFoundError(
        f"AMP clips missing in {motion_dir}: {missing} (convert motion_data_csv/amp "
        "with AMP_mjlab's scripts/csv_to_npz.py, or set AME_AMP_MOTION_DIR)"
      )
    self.names = sorted(clips)
    self.frames: list[torch.Tensor] = []
    for c in self.names:
      d = np.load(motion_dir / f"{c}.npz")
      if int(d["fps"][0]) != 50:
        raise ValueError(f"{c}: fps {d['fps']} != 50 (policy dt 0.02 s)")
      pos, quat = (torch.tensor(d[k], dtype=torch.float32) for k in ("body_pos_w", "body_quat_w"))
      lin, ang = (torch.tensor(d[k], dtype=torch.float32) for k in ("body_lin_vel_w", "body_ang_vel_w"))
      if pos.shape[1] != len(names):
        raise ValueError(f"{c}: {pos.shape[1]} bodies in the npz, robot has {len(names)}")
      obs = amp_observation(
        pos[:, body_ids], quat[:, body_ids], lin[:, body_ids], ang[:, body_ids],
        pos[:, anchor], quat[:, anchor],
      )
      self.frames.append(obs.to(device))
    self.dim = self.frames[0].shape[1]
    self.device = device

  def feed_forward_generator(
    self, num_mini_batch: int, mini_batch_size: int
  ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for b in range(num_mini_batch):
      f = self.frames[b % len(self.frames)]
      t = f.shape[0]
      idx = torch.randint(0, t, (mini_batch_size,), device=self.device)
      yield f[idx], f[torch.clamp(idx + 1, max=t - 1)]


# ------------------------------------------------------------------ modules
class Discriminator(nn.Module):
  def __init__(self, input_dim: int, hidden_dims: Sequence[int] = AMP_DISCR_HIDDEN_DIMS):
    super().__init__()
    layers: list[nn.Module] = []
    d = input_dim
    for h in hidden_dims:
      layers += [nn.Linear(d, h), nn.ReLU()]
      d = h
    self.input_dim = input_dim
    self.trunk = nn.Sequential(*layers)
    self.amp_linear = nn.Linear(d, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.amp_linear(self.trunk(x))

  def grad_pen(self, s: torch.Tensor, s_next: torch.Tensor, lambda_: float) -> torch.Tensor:
    x = torch.cat([s, s_next], dim=-1).detach().requires_grad_(True)
    d = self(x)
    (g,) = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True, retain_graph=True)
    return lambda_ * g.norm(2, dim=1).pow(2).mean()


class AmpNormalizer(nn.Module):
  """Running mean / var (float64), the reference's ``Normalizer``."""

  def __init__(self, dim: int, epsilon: float = 1e-4, clip: float = 10.0) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
    self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
    self.epsilon, self.clip = epsilon, clip

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.detach().double()
    bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = bm - self.mean
    tot = self.count + bc
    m2 = self.var * self.count + bv * bc + delta.square() * self.count * bc / tot
    self.mean.add_(delta * bc / tot)
    self.var.copy_(m2 / tot)
    self.count.fill_(tot.item())

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(self.var + self.epsilon).float()
    return torch.clamp((x - self.mean.float()) / std, -self.clip, self.clip)


class ReplayBuffer:
  def __init__(self, dim: int, size: int, device: str) -> None:
    self.states = torch.zeros(size, dim, device=device)
    self.next_states = torch.zeros(size, dim, device=device)
    self.size, self.step, self.num_samples = size, 0, 0

  def insert(self, s: torch.Tensor, s_next: torch.Tensor) -> None:
    n = s.shape[0]
    idx = (self.step + torch.arange(n, device=s.device)) % self.size
    self.states[idx] = s
    self.next_states[idx] = s_next
    self.num_samples = min(self.size, self.num_samples + n)
    self.step = (self.step + n) % self.size

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    for _ in range(num_mini_batch):
      idx = torch.randint(0, self.num_samples, (mini_batch_size,), device=self.states.device)
      yield self.states[idx], self.next_states[idx]


# ------------------------------------------------------------------ algorithm
class AmpDoubleCriticPPO(DoubleCriticPPO):
  """:class:`DoubleCriticPPO` + AMP (see the module docstring)."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    amp_motion_dir: str = AMP_MOTION_DIR,
    amp_clips: Sequence[str] = AMP_CLIPS,
    amp_body_names: Sequence[str] = AMP_BODY_NAMES,
    amp_anchor_name: str = AMP_ANCHOR_NAME,
    amp_reward_coef: float = AMP_REWARD_COEF,
    amp_task_reward_lerp: float = AMP_TASK_REWARD_LERP,
    amp_discr_hidden_dims: Sequence[int] = AMP_DISCR_HIDDEN_DIMS,
    amp_replay_buffer_size: int = AMP_REPLAY_BUFFER_SIZE,
    amp_grad_pen_lambda: float = AMP_GRAD_PEN_LAMBDA,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    self.amp_motion_dir = amp_motion_dir
    self.amp_clips = tuple(amp_clips)
    self.amp_body_names = tuple(amp_body_names)
    self.amp_anchor_name = amp_anchor_name
    self.amp_reward_coef = float(amp_reward_coef)
    self.amp_task_reward_lerp = float(amp_task_reward_lerp)
    self.amp_grad_pen_lambda = float(amp_grad_pen_lambda)
    dim = 15 * len(self.amp_body_names)
    self.discriminator = Discriminator(2 * dim, amp_discr_hidden_dims).to(self.device)
    self.amp_normalizer = AmpNormalizer(dim).to(self.device)
    self.amp_buffer = ReplayBuffer(dim, int(amp_replay_buffer_size), self.device)
    self.optimizer.add_param_group(
      {"params": self.discriminator.trunk.parameters(), "weight_decay": AMP_TRUNK_WEIGHT_DECAY}
    )
    self.optimizer.add_param_group(
      {"params": self.discriminator.amp_linear.parameters(), "weight_decay": AMP_HEAD_WEIGHT_DECAY}
    )
    self.motion: MotionData | None = None
    self._robot = None
    self._amp_obs: torch.Tensor | None = None
    self._style: torch.Tensor | None = None
    self._rollout = {"style": 0.0, "d_policy_rollout": 0.0, "n": 0}

  # -- binding -----------------------------------------------------------------
  def bind_reward_groups(self, env) -> None:
    super().bind_reward_groups(env)
    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene["robot"]
    names = list(robot.body_names)
    self._robot = robot
    self._body_ids = torch.tensor([names.index(n) for n in self.amp_body_names], device=self.device)
    self._anchor_id = names.index(self.amp_anchor_name)
    self.motion = MotionData(
      resolve_motion_dir(self.amp_motion_dir), self.amp_clips, names,
      self.amp_body_names, self.amp_anchor_name, self.device,
    )
    if self.motion.dim != self.amp_buffer.states.shape[1]:
      raise ValueError(f"AMP dim {self.motion.dim} != {self.amp_buffer.states.shape[1]}")

  def robot_amp_obs(self) -> torch.Tensor:
    d = self._robot.data
    ids = self._body_ids
    return amp_observation(
      d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids],
      d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids],
      d.body_link_pos_w[:, self._anchor_id], d.body_link_quat_w[:, self._anchor_id],
    ).to(self.device)

  # -- rollout -----------------------------------------------------------------
  def act(self, obs):
    self._amp_obs = self.robot_amp_obs()
    return super().act(obs)

  def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      n = self.amp_normalizer
      d = self.discriminator(torch.cat([n.normalize(s), n.normalize(s_next)], dim=-1))
      r = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (d - 1.0).square(), min=0.0)
    return r.squeeze(-1), d.squeeze(-1)

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    assert self._amp_obs is not None, "act() must run before process_env_step()"
    nxt = self.robot_amp_obs()
    done = dones.to(self.device).bool()
    nxt = torch.where(done[:, None], self._amp_obs, nxt)
    self._style, d = self.style_reward(self._amp_obs, nxt)
    self.amp_buffer.insert(self._amp_obs, nxt)
    self._rollout["style"] += self._style.mean().item()
    self._rollout["d_policy_rollout"] += d.mean().item()
    self._rollout["n"] += 1
    super().process_env_step(obs, rewards, dones, extras)

  def group_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
    g = super().group_rewards(rewards)
    lerp = self.amp_task_reward_lerp
    g1 = (1.0 - lerp) * self._style.to(g.device) + lerp * g[:, 0]
    return torch.stack([g1, g[:, 1]], dim=-1)

  # -- update ------------------------------------------------------------------
  def update(self) -> dict[str, float]:
    n_mb = self.num_learning_epochs * self.num_mini_batches
    mbs = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
    policy_gen = self.amp_buffer.feed_forward_generator(n_mb, mbs)
    expert_gen = self.motion.feed_forward_generator(n_mb, mbs)
    acc = {"amp_loss": 0.0, "amp_grad_pen": 0.0, "amp_d_policy": 0.0, "amp_d_expert": 0.0}
    count = [0]
    optimizer = self.optimizer
    if "step" in vars(optimizer):
      raise RuntimeError("optimizer.step is already patched")
    step = optimizer.step

    def amp_step(*args, **kwargs):
      # PPO.update has just zeroed every group's gradient and back-propagated
      # the PPO loss; add the discriminator's before the shared Adam step.
      ps, pns = next(policy_gen)
      es, ens = next(expert_gen)
      n = self.amp_normalizer
      with torch.no_grad():
        ps_n, pns_n, es_n, ens_n = (n.normalize(x) for x in (ps, pns, es, ens))
      d_p = self.discriminator(torch.cat([ps_n, pns_n], dim=-1))
      d_e = self.discriminator(torch.cat([es_n, ens_n], dim=-1))
      amp_loss = 0.5 * (
        nn.functional.mse_loss(d_e, torch.ones_like(d_e))
        + nn.functional.mse_loss(d_p, -torch.ones_like(d_p))
      )
      gp = self.discriminator.grad_pen(es, ens, self.amp_grad_pen_lambda)
      (amp_loss + gp).backward()
      out = step(*args, **kwargs)
      n.update(ps)
      n.update(es)
      acc["amp_loss"] += amp_loss.item()
      acc["amp_grad_pen"] += gp.item()
      acc["amp_d_policy"] += d_p.mean().item()
      acc["amp_d_expert"] += d_e.mean().item()
      count[0] += 1
      return out

    optimizer.step = amp_step
    try:
      losses = super().update()
    finally:
      del optimizer.step
    if count[0] != n_mb:
      raise RuntimeError(f"discriminator stepped {count[0]} times, expected {n_mb}")
    for k, v in acc.items():
      losses[k] = v / count[0]
    r = self._rollout
    if r["n"]:
      losses["amp_style_reward"] = r["style"] / r["n"]
      losses["amp_d_policy_rollout"] = r["d_policy_rollout"] / r["n"]
    self._rollout = {"style": 0.0, "d_policy_rollout": 0.0, "n": 0}
    return losses

  # -- checkpoint ----------------------------------------------------------------
  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def save(self) -> dict:
    saved = super().save()
    saved["discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    out = super().load(loaded_dict, load_cfg, strict)
    if (load_cfg is None or load_cfg.get("discriminator", load_cfg.get("optimizer"))) and (
      "discriminator_state_dict" in loaded_dict
    ):
      self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
      self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
    return out


__all__ = [
  "AMP_ANCHOR_NAME",
  "AMP_BODY_NAMES",
  "AMP_CLIPS",
  "AMP_MOTION_DIR",
  "AmpDoubleCriticPPO",
  "AmpNormalizer",
  "Discriminator",
  "MotionData",
  "ReplayBuffer",
  "amp_observation",
]
