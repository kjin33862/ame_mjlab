"""Double-critic PPO (BeamDojo, arXiv 2502.10363, Sec. IV-B / Table X) on rsl_rl 5.4.

Two reward groups, each with its own value head and its own GAE:

* group 1: every reward term except the group-2 terms (the existing stack);
* group 2: the group-2 terms (here the ``foothold`` reward).

Each group's advantage is normalised by its own batch mean / std, and the policy
gradient uses ``A = w1 * A1_n + w2 * A2_n`` (``advantage_weights``, BeamDojo's
1.0 / 0.25).  The value loss is the sum of the two heads' losses.

How the env delivers two groups.  mjlab's ``RewardManager.compute`` returns one
summed tensor, but it also keeps every term's weighted, unscaled value of the step
in ``_step_reward`` (``[B, num_terms]``, NaN-cleaned exactly like the sum).  So the
env and the reward manager stay untouched: the reward manager computes all terms
(group 2 included, so ``Episode_Reward/foothold`` is still logged), and
:class:`RewardGroups` reads the group-2 columns right after ``env.step`` and returns
``[r_total - r2, r2]``.  ``construct_algorithm`` receives the env, which is where the
reader is bound.  rsl_rl's runner already passes the step's reward to
``alg.process_env_step``, so no runner change is needed; the runner cfg names this
algorithm class.

The critic is one network with two outputs (two value heads on a shared trunk and,
with ``share_encoder``, the shared terrain encoder).  A double-critic checkpoint's
critic does not load into a single-critic runner (and vice versa); the actor does.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

ADVANTAGE_WEIGHTS = (1.0, 0.25)
"""BeamDojo Table X: w1 for the dense group, w2 for the foothold group."""


# ------------------------------------------------------------------ reward split
class RewardGroups:
  """Split the env's summed reward into ``[group 1, group 2]``.

  ``reward_manager``: mjlab's ``RewardManager``; ``group2_terms``: names of the
  group-2 terms (all must be active); ``dt``: the env step (the manager scales its
  sum by dt unless ``scale_by_dt`` is off)."""

  def __init__(self, reward_manager, group2_terms: Sequence[str], dt: float):
    names = list(reward_manager.active_terms)
    missing = [t for t in group2_terms if t not in names]
    if missing or not group2_terms:
      raise ValueError(
        f"group-2 reward terms {missing or list(group2_terms)} not in the reward "
        f"manager's active terms {names}"
      )
    self.manager = reward_manager
    self.index = torch.tensor([names.index(t) for t in group2_terms], dtype=torch.long)
    self.scale = dt if getattr(reward_manager, "_scale_by_dt", True) else 1.0

  def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
    step = self.manager._step_reward  # [B, T] weight * value, not dt-scaled
    r2 = step[:, self.index.to(step.device)].sum(dim=1).to(rewards.device) * self.scale
    return torch.stack([rewards - r2, r2], dim=-1)


# ------------------------------------------------------------------ advantages
def gae(
  rewards: torch.Tensor,
  values: torch.Tensor,
  dones: torch.Tensor,
  last_values: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """Returns ``[T, N, G]`` (rsl_rl's recursion, per channel).  ``rewards`` /
  ``values``: ``[T, N, G]``; ``dones``: ``[T, N, 1]``; ``last_values``: ``[N, G]``."""
  T = rewards.shape[0]
  returns = torch.zeros_like(values)
  advantage = torch.zeros_like(last_values)
  for step in reversed(range(T)):
    next_values = last_values if step == T - 1 else values[step + 1]
    not_terminal = 1.0 - dones[step].float()
    delta = rewards[step] + not_terminal * gamma * next_values - values[step]
    advantage = delta + not_terminal * gamma * lam * advantage
    returns[step] = advantage + values[step]
  return returns


def combine_advantages(
  advantages: torch.Tensor, weights: Sequence[float], eps: float = 1e-8
) -> torch.Tensor:
  """``[T, N, G]`` -> ``[T, N, 1]``: each channel normalised by its own batch
  mean / std (rsl_rl's formula), then weighted and summed."""
  flat = advantages.reshape(-1, advantages.shape[-1])
  normed = (advantages - flat.mean(dim=0)) / (flat.std(dim=0) + eps)
  w = torch.as_tensor(weights, dtype=advantages.dtype, device=advantages.device)
  return (normed * w).sum(dim=-1, keepdim=True)


# ------------------------------------------------------------------ storage
class MultiRewardRolloutStorage(RolloutStorage):
  """``RolloutStorage`` with ``G`` reward / value / return channels; the
  advantage stays one channel (the combined one)."""

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape,
    device: str = "cpu",
    num_groups: int = 2,
  ) -> None:
    super().__init__(
      training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
    )
    self.num_groups = num_groups
    shape = (num_transitions_per_env, num_envs, num_groups)
    self.rewards = torch.zeros(*shape, device=self.device)
    self.values = torch.zeros(*shape, device=self.device)
    self.returns = torch.zeros(*shape, device=self.device)

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    rewards = transition.rewards
    assert rewards is not None and rewards.shape[-1] == self.num_groups
    # The base copies rewards as [N, 1]; hand it a placeholder and write ours.
    transition.rewards = rewards[..., :1]
    step = self.step
    super().add_transition(transition)
    self.rewards[step].copy_(rewards)
    transition.rewards = rewards


# ------------------------------------------------------------------ algorithm
class DoubleCriticPPO(PPO):
  """PPO with one value head per reward group (see the module docstring)."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    reward_group_terms: Sequence[str] = ("foothold",),
    advantage_weights: Sequence[float] = ADVANTAGE_WEIGHTS,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if self.rnd is not None:
      raise ValueError("DoubleCriticPPO does not support RND")
    if self.normalize_advantage_per_mini_batch:
      raise ValueError(
        "DoubleCriticPPO normalises each group over the whole batch; "
        "normalize_advantage_per_mini_batch must be False"
      )
    if len(advantage_weights) != 2:
      raise ValueError("two reward groups: advantage_weights must have 2 entries")
    self.reward_group_terms = tuple(reward_group_terms)
    self.advantage_weights = tuple(float(w) for w in advantage_weights)
    self.num_groups = 2
    # PPO.update averages the elementwise value loss over [B, G]; scaling the
    # coefficient by G makes it the sum of the per-head losses.
    self.value_loss_coef = self.value_loss_coef * self.num_groups
    self.reward_groups: RewardGroups | None = None
    self._stats: dict[str, float] = {}

  def bind_reward_groups(self, env) -> None:
    unwrapped = getattr(env, "unwrapped", env)
    self.reward_groups = RewardGroups(
      unwrapped.reward_manager, self.reward_group_terms, unwrapped.step_dt
    )

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    if self.reward_groups is None:
      raise RuntimeError("bind_reward_groups(env) was not called")
    super().process_env_step(obs, self.reward_groups(rewards), dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    st = self.storage
    hidden = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    self.critic.reset(hidden_state=hidden)
    st.returns = gae(st.rewards, st.values, st.dones, last_values, self.gamma, self.lam)
    raw = st.returns - st.values
    st.advantages = combine_advantages(raw, self.advantage_weights)
    flat_raw = raw.reshape(-1, self.num_groups)
    flat_ret = st.returns.reshape(-1, self.num_groups)
    flat_val = st.values.reshape(-1, self.num_groups)
    for g in range(self.num_groups):
      var = flat_ret[:, g].var()
      ev = 1.0 - (flat_ret[:, g] - flat_val[:, g]).var() / (var + 1e-8)
      self._stats[f"adv_std_g{g + 1}"] = flat_raw[:, g].std().item()
      self._stats[f"return_mean_g{g + 1}"] = flat_ret[:, g].mean().item()
      self._stats[f"explained_var_g{g + 1}"] = ev.item()

  def update(self) -> dict[str, float]:
    losses = super().update()
    losses.update(self._stats)
    return losses

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
    """rsl_rl's ``PPO.construct_algorithm`` with a two-output critic, the
    multi-reward storage, and the reward split bound to ``env``."""
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
    actor = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    )
    actor = actor.to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic = critic_class(obs, cfg["obs_groups"], "critic", 2, **cfg["critic"]).to(
      device
    )
    print(f"Critic Model (2 value heads): {critic}")
    storage = MultiRewardRolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device, 2
    )
    alg = alg_class(
      actor,
      critic,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.bind_reward_groups(env)
    alg.compile(cfg.get("torch_compile_mode"))
    return alg


__all__ = [
  "ADVANTAGE_WEIGHTS",
  "DoubleCriticPPO",
  "MultiRewardRolloutStorage",
  "RewardGroups",
  "combine_advantages",
  "gae",
]
