"""AME-2-style goal reaching on the gap board, plus the velocity stagnation guard.

AME-2 (arXiv 2601.08485; official code ``leggedrobotics/ame2_minimal``) trains
on one position + heading goal per episode instead of a velocity command.  The
pieces here follow the official terms (file / line in notes/goal_reach/report.md):

* :class:`GoalCommandCfg` / :class:`GoalCommand` — one goal per episode near
  the robot's tile centre, the command ``(x_b, y_b, z_b, heading_error)`` in the
  robot's yaw frame (``TerrainBasedPose2dCommand``'s layout);
* :class:`GoalGaitCommandCfg` — a derived, never-observed twist that gates the
  base task's command-gated regularisers (feet clearance, swing height, slip,
  soft landing, posture regime) by "still walking to the goal" vs "at the goal";
* :func:`goal_obs` / :func:`time_left` — ``ame2_commands_obs`` (local goal xy
  clipped to 2 m for the actor, heading error as sin / cos, randomised while
  far) and ``ame2_time_left`` (critic only);
* :func:`goal_position_tracking`, :func:`goal_heading_tracking`,
  :func:`goal_move_to_goal`, :func:`goal_stand_at_goal` — the four task
  rewards;
* :class:`goal_stagnation` — ``ame2_stagnation``;
* :class:`velocity_stagnation` — a velocity-task guard: the command asked for
  > 0.3 m/s for the last 5 s and the pelvis moved < 0.5 m;
* :class:`terrain_levels_sgs_goal` — SGS with AME-2's goal-reached success
  (``ame2_terrain_levels_goalreaching``: within 0.5 m with < 4 s left) and the
  goal-task episode logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from ame.mdp.sgs import OutcomeBuffer, terrain_levels_sgs

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

GOAL_COMMAND = "goal"
GAIT_COMMAND = "gait"

GOAL_OBS_CLIP = 2.0
"""Actor goal-distance clip, m (official ``dist_thr=2.0``)."""

GOAL_SCHEDULE_STEPS = 24 * 2000
"""Official curriculum length in env steps: 2000 iterations of 24 steps."""

GOAL_YAW_SCHEDULE = {"steps": GOAL_SCHEDULE_STEPS, "limit": math.pi}
"""Spawn-yaw curriculum: the range grows from 0 to +/- ``limit`` over
``steps`` env steps (official ``ame2_modify_yaw_range``)."""

STAIR_FAMILIES = ("pyramid_stairs", "pyramid_stairs_inv")


def linear_schedule(counter: int, steps: int, start: float, end: float) -> float:
  """``start`` -> ``end`` linearly over ``steps`` env steps, then held."""
  progress = min(float(counter) / max(int(steps), 1), 1.0)
  return start + progress * (end - start)


def _robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> Any:
  return env.scene[asset_cfg.name]


def time_left_s(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Remaining episode time, s (official ``(max_len - buf) * step_dt``)."""
  return (env.max_episode_length - env.episode_length_buf).float() * env.step_dt


# --------------------------------------------------------------------------
# Commands.
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
  """One goal per episode around the tile centre.

  Direction: within +/- ``cone`` of world +x (the env frame), the cone
  growing from ``cone_start_deg`` to ``cone_end_deg`` over ``cone_steps``
  env steps; families in ``straight_families`` (the stairs) go straight
  along +x with ``straight_lateral`` m of lateral jitter, as the official
  stair patches (y in +/- 0.2).  Distance from the tile centre: uniform in
  ``distance_range`` (per family overrides in ``family_distance_range``),
  then scaled so the Chebyshev distance stays within ``max_chebyshev``.  On a
  stone lattice the goal is the nearest stone centre (``snap_to_stones``).
  Heading: uniform in ``heading_range`` in the world frame.
  """

  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  distance_range: tuple[float, float] = (1.0, 3.5)
  family_distance_range: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {name: (2.0, 3.5) for name in STAIR_FAMILIES}
  )
  straight_families: tuple[str, ...] = STAIR_FAMILIES
  straight_lateral: float = 0.2
  max_chebyshev: float = 3.5
  heading_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)
  cone_start_deg: float = 10.0
  cone_end_deg: float = 45.0
  cone_steps: int = GOAL_SCHEDULE_STEPS
  snap_to_stones: bool = True

  def build(self, env: ManagerBasedRlEnv) -> GoalCommand:
    return GoalCommand(self, env)


def lattice_stone_table(terrain: Any) -> tuple[torch.Tensor, torch.Tensor]:
  """Per (row, col) stone centres in world xy, padded with +inf, and a
  per-column "is a lattice" mask (the centre spawn's construction)."""
  from ame.terrains import BoxStoneLatticeTerrainCfg

  gen = terrain.cfg.terrain_generator
  origins = terrain.terrain_origins.cpu().numpy()
  rows, cols = origins.shape[:2]
  subs = list(gen.sub_terrains.values())
  device = terrain.terrain_origins.device
  lower, upper = gen.difficulty_range
  is_lattice = [isinstance(s, BoxStoneLatticeTerrainCfg) for s in subs]
  tables: dict[tuple[int, int], np.ndarray] = {}
  for c, sub in enumerate(subs):
    if is_lattice[c]:
      for r in range(rows):
        difficulty = lower + (upper - lower) * r / max(rows - 1, 1)
        grid, _ = sub.centres(difficulty)
        tables[r, c] = grid - np.asarray(sub.size) / 2 + origins[r, c, :2]
  k = max((len(t) for t in tables.values()), default=1)
  stones = np.full((rows, cols, k, 2), np.inf)
  for (r, c), t in tables.items():
    stones[r, c, : len(t)] = t
  return (
    torch.tensor(stones, dtype=torch.float32, device=device),
    torch.tensor(is_lattice, device=device),
  )


class GoalCommand(CommandTerm):
  """Goal position / heading; ``command`` is ``(x_b, y_b, z_b, heading_err)``
  with the position in the robot's yaw frame and the heading error
  ``wrap(heading_goal - heading_robot)``."""

  cfg: GoalCommandCfg

  def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_w = torch.zeros(self.num_envs, device=self.device)
    self.goal_b = torch.zeros(self.num_envs, 4, device=self.device)
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("the goal command needs a generated terrain")
    gen = terrain.cfg.terrain_generator
    if gen is None or not gen.curriculum:
      raise ValueError("the goal command needs a curriculum board")
    names = list(gen.sub_terrains.keys())
    if terrain.terrain_origins.shape[1] != len(names):
      raise ValueError("a curriculum board has one column per family")
    ranges = [cfg.family_distance_range.get(n, cfg.distance_range) for n in names]
    self.distance_range = torch.tensor(ranges, dtype=torch.float32, device=self.device)
    self.straight = torch.tensor(
      [n in cfg.straight_families for n in names], device=self.device
    )
    self.stones, self.is_lattice = lattice_stone_table(terrain)
    self.root_height = float(self.robot.data.default_root_state[0, 2].item())

  @property
  def command(self) -> torch.Tensor:
    return self.goal_b

  def cone_deg(self) -> float:
    c = self.cfg
    return linear_schedule(
      self._env.common_step_counter, c.cone_steps, c.cone_start_deg, c.cone_end_deg
    )

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    c = self.cfg
    terrain: Any = self._env.scene.terrain
    n = len(env_ids)
    dev = self.device
    level = terrain.terrain_levels[env_ids]
    family = terrain.terrain_types[env_ids]
    centre = terrain.terrain_origins[level, family]
    lo, hi = self.distance_range[family, 0], self.distance_range[family, 1]
    dist = lo + torch.rand(n, device=dev) * (hi - lo)
    cone = math.radians(self.cone_deg())
    theta = (2 * torch.rand(n, device=dev) - 1) * cone
    straight = self.straight[family]
    theta = torch.where(straight, torch.zeros_like(theta), theta)
    off = torch.stack([dist * torch.cos(theta), dist * torch.sin(theta)], dim=1)
    lateral = (2 * torch.rand(n, device=dev) - 1) * c.straight_lateral
    off[:, 1] += torch.where(straight, lateral, torch.zeros_like(lateral))
    cheb = off.abs().amax(dim=1).clamp(min=1e-6)
    off *= (c.max_chebyshev / cheb).clamp(max=1.0).unsqueeze(1)
    xy = centre[:, :2] + off
    if c.snap_to_stones:
      on_lattice = self.is_lattice[family]
      if bool(on_lattice.any()):
        stones = self.stones[level, family]  # [n, K, 2]
        nearest = torch.linalg.norm(stones - xy[:, None, :], dim=-1).argmin(dim=1)
        stone = stones[torch.arange(n, device=dev), nearest]
        xy = torch.where(on_lattice[:, None], stone, xy)
    self.pos_w[env_ids, :2] = xy
    self.pos_w[env_ids, 2] = centre[:, 2] + self.root_height
    self.heading_w[env_ids] = torch.empty(n, device=dev).uniform_(*c.heading_range)
    self._update_command()

  def _update_command(self) -> None:
    data = self.robot.data
    quat = yaw_quat(data.root_link_quat_w)
    self.goal_b[:, :3] = quat_apply_inverse(quat, self.pos_w - data.root_link_pos_w)
    self.goal_b[:, 3] = wrap_to_pi(self.heading_w - data.heading_w)


@dataclass(kw_only=True)
class GoalGaitCommandCfg(CommandTermCfg):
  """A twist derived from the goal command, never observed: the goal
  direction at ``walk_speed`` while farther than ``at_goal_distance``, else
  zero; inside it the yaw rate is the heading error clipped to
  ``+/- turn_rate``.  Its norm is 0 (standing regime) or between 0.05 and
  1.5 (walking regime) under the base task's thresholds, so the
  command-gated regularisers stay on while walking to the goal and the
  posture term holds the standing pose once there."""

  goal_command: str = GOAL_COMMAND
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  walk_speed: float = 1.0
  at_goal_distance: float = 0.5
  turn_rate: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> GoalGaitCommand:
    return GoalGaitCommand(self, env)


class GoalGaitCommand(CommandTerm):
  cfg: GoalGaitCommandCfg

  def __init__(self, cfg: GoalGaitCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.twist = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.twist

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids
    self._update_command()

  def _update_command(self) -> None:
    c = self.cfg
    goal = self._env.command_manager.get_term(c.goal_command)
    if goal is None:
      return
    g = goal.command
    xy = g[:, :2]
    dist = torch.norm(xy, dim=1, keepdim=True)
    far = dist > c.at_goal_distance
    self.twist[:, :2] = torch.where(
      far, c.walk_speed * xy / dist.clamp(min=1e-6), torch.zeros_like(xy)
    )
    self.twist[:, 2] = torch.where(
      far.squeeze(1),
      torch.zeros_like(g[:, 3]),
      g[:, 3].clamp(-c.turn_rate, c.turn_rate),
    )


# --------------------------------------------------------------------------
# Observations.
# --------------------------------------------------------------------------


def goal_obs(
  env: ManagerBasedRlEnv,
  command_name: str = GOAL_COMMAND,
  dist_clip: float | None = None,
) -> torch.Tensor:
  """``[x, y, sin(e), cos(e)]`` of the goal in the yaw frame (official
  ``ame2_commands_obs``).  With ``dist_clip``, a goal farther than that is
  shown at ``dist_clip`` along its direction and its heading error ``e`` is
  replaced by a uniform random angle (re-drawn every step)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  pos = command[:, :2]
  err = command[:, 3]
  if dist_clip is not None:
    dist = torch.norm(pos, dim=1)
    far = dist > dist_clip
    scaled = pos * (dist_clip / dist.clamp(min=1e-6)).unsqueeze(1)
    pos = torch.where(far.unsqueeze(1), scaled, pos)
    random_err = (torch.rand_like(err) * 2 - 1) * math.pi
    err = torch.where(far, random_err, err)
  return torch.cat([pos, torch.sin(err)[:, None], torch.cos(err)[:, None]], dim=1)


def time_left(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Normalised remaining episode time ``1 - t / T`` (official
  ``ame2_time_left``)."""
  return (1.0 - env.episode_length_buf.float() / env.max_episode_length).unsqueeze(1)


# --------------------------------------------------------------------------
# Rewards.
# --------------------------------------------------------------------------


def goal_position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = GOAL_COMMAND,
  T: float = 4.0,
  std: float = 2.0,
) -> torch.Tensor:
  """``1 / (1 + (d_xy / std)^2) * 1(t_left < T) / T``."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  d_xy = torch.norm(command[:, :2], dim=1)
  mask = (time_left_s(env) < T).float() / T
  return mask / (1.0 + torch.square(d_xy / std))


def goal_heading_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = GOAL_COMMAND,
  T: float = 2.0,
  dist_thr: float = 0.5,
) -> torch.Tensor:
  """``1 / (1 + e_yaw^2) * 1(t_left < T) / T * 1(d_xy < dist_thr)``."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  d_xy = torch.norm(command[:, :2], dim=1)
  mask = (time_left_s(env) < T).float() / T
  near = (d_xy < dist_thr).float()
  return mask * near / (1.0 + torch.square(command[:, 3]))


def goal_move_to_goal(
  env: ManagerBasedRlEnv,
  command_name: str = GOAL_COMMAND,
  v_min: float = 0.3,
  v_max: float = 2.0,
  cos_thr: float = 0.5,
  dist_thr: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """1 if within ``dist_thr`` of the goal, or moving toward it (cosine of
  the velocity to the goal direction > ``cos_thr``) at ``v_min`` .. ``v_max``
  m/s; the velocity is ``asset_cfg``'s body (the torso), rotated into the
  root frame and taken in xy (official ``ame2_move2goal``)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  goal_xy = command[:, :2]
  dist = torch.norm(goal_xy, dim=1)
  asset = _robot(env, asset_cfg)
  ids = asset_cfg.body_ids
  if ids is None or isinstance(ids, slice):
    vel_w = asset.data.root_link_lin_vel_w[:, :3]
  else:
    idx = ids[0] if not isinstance(ids, int) else ids
    vel_w = asset.data.body_link_lin_vel_w[:, idx, :3]
  vel_xy = quat_apply_inverse(asset.data.root_link_quat_w, vel_w)[:, :2]
  speed = torch.norm(vel_xy, dim=1)
  cos = (vel_xy * goal_xy).sum(dim=1) / (speed * dist + 1e-6)
  toward = (cos > cos_thr) & (speed >= v_min) & (speed <= v_max)
  return ((dist < dist_thr) | toward).float()


def goal_stand_at_goal(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = GOAL_COMMAND,
  dist_thr: float = 0.5,
  yaw_thr: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``1(d_xy < dist_thr and |e_yaw| < yaw_thr) * exp(-(d_foot + d_g + d_q +
  d_xy) / 4)``: ``d_foot`` the share of feet out of contact, ``d_g = 1 -
  g_z^2``, ``d_q`` the mean |joint - default| (official ``ame2_standatgoal``)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  d_xy = torch.norm(command[:, :2], dim=1)
  at_goal = (d_xy < dist_thr) & (command[:, 3].abs() < yaw_thr)
  found = env.scene[sensor_name].data.found
  assert found is not None
  in_contact = (found.reshape(found.shape[0], -1) > 0).float()
  d_foot = 1.0 - in_contact.mean(dim=1)
  asset = _robot(env, asset_cfg)
  d_g = 1.0 - torch.square(asset.data.projected_gravity_b[:, 2])
  d_q = (asset.data.joint_pos - asset.data.default_joint_pos).abs().mean(dim=1)
  return at_goal.float() * torch.exp(-(d_foot + d_g + d_q + d_xy) / 4.0)


# --------------------------------------------------------------------------
# Terminations.
# --------------------------------------------------------------------------


class goal_stagnation:
  """Official ``ame2_stagnation``: every ``time_window`` s the root position
  is compared with the last snapshot; stagnant if it moved (3-D) less than
  ``distance_threshold`` while the goal is more than
  ``goal_distance_threshold`` away.  The snapshot refreshes at the episode's
  first step and at every check.  The window is counted in env steps."""

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.last_pos = torch.zeros(env.num_envs, 3, device=env.device)
    self.last_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = GOAL_COMMAND,
    distance_threshold: float = 0.5,
    goal_distance_threshold: float = 1.0,
    time_window: float = 5.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
  ) -> torch.Tensor:
    pos = _robot(env, asset_cfg).data.root_link_pos_w
    step = env.episode_length_buf.long()
    fresh = step <= 1
    self.last_pos[fresh] = pos[fresh]
    self.last_step[fresh] = step[fresh]
    window = round(time_window / env.step_dt)
    check = (step - self.last_step) >= window
    moved = torch.norm(pos - self.last_pos, dim=1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    far = torch.norm(command[:, :2], dim=1) > goal_distance_threshold
    stagnant = check & (moved < distance_threshold) & far
    self.last_pos[check] = pos[check]
    self.last_step[check] = step[check]
    return stagnant

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # the first step of the next episode refreshes the snapshot


class velocity_stagnation:
  """A velocity-task stall guard: terminate when the command's xy speed has
  been above ``command_threshold`` for the last ``time_window`` s and the
  pelvis's horizontal displacement over that window is below
  ``distance_threshold``.  A rolling window: a ring of the last
  ``time_window / step_dt`` xy positions; any lower command restarts the
  count, so a standing command never terminates."""

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    window = round(float(cfg.params.get("time_window", 5.0)) / env.step_dt)
    self.window = int(window)
    self.ring = torch.zeros(self.window, env.num_envs, 2, device=env.device)
    self.ptr = 0
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.3,
    distance_threshold: float = 0.5,
    time_window: float = 5.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
  ) -> torch.Tensor:
    del time_window  # resolved at construction
    xy = _robot(env, asset_cfg).data.root_link_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    moving = torch.norm(command[:, :2], dim=1) > command_threshold
    fresh = env.episode_length_buf <= 1
    self.count = torch.where(
      moving & ~fresh, self.count + 1, torch.zeros_like(self.count)
    )
    old = self.ring[self.ptr]  # the position ``window`` steps ago
    # ``count > window``: the old sample was written under a moving command
    # in this episode.
    stalled = (self.count > self.window) & (
      torch.norm(xy - old, dim=1) < distance_threshold
    )
    self.ring[self.ptr] = xy
    self.ptr = (self.ptr + 1) % self.window
    return stalled

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.count[slice(None) if env_ids is None else env_ids] = 0


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------


def goal_distance(
  env: ManagerBasedRlEnv, command_name: str = GOAL_COMMAND
) -> torch.Tensor:
  """Horizontal distance to the goal, m (``reduce="last"``: the final one)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1)


# --------------------------------------------------------------------------
# Curriculum.
# --------------------------------------------------------------------------

GOAL_LOG_EPISODES = 1000
"""Window of the goal-task episode logs, episodes (overall and per family)."""


class _FloatRing:
  def __init__(self, size: int, device: Any) -> None:
    self.values = torch.zeros(size, device=device)
    self.head = 0
    self.fill = 0

  def record(self, x: torch.Tensor) -> None:
    x = x.flatten()[-self.values.numel() :].float()
    n = x.numel()
    if n == 0:
      return
    idx = (self.head + torch.arange(n, device=x.device)) % self.values.numel()
    self.values[idx] = x
    self.head = int((self.head + n) % self.values.numel())
    self.fill = min(self.fill + n, self.values.numel())

  def mean(self) -> float:
    return float(self.values[: self.fill].mean().item()) if self.fill else 0.0


class terrain_levels_sgs_goal(terrain_levels_sgs):
  """SGS whose success is AME-2's goal reached: at the episode end the
  pelvis is within ``reach_distance`` of the goal (xy) with less than
  ``reach_time_left`` s left (official ``ame2_terrain_levels_goalreaching``:
  ``dist < 0.5 and t_left < 4``).  Sampling, buffers, floor and arming are
  :class:`terrain_levels_sgs`'s (no EMA gate, no promotion ladder).

  Logs, over the last ``GOAL_LOG_EPISODES`` ended episodes: ``goal/reached``,
  ``goal/final_distance``, ``goal/stagnation``, ``goal/contact``
  (``illegal_contact``), ``goal/fell``, ``goal/timeout``, per family
  ``<family>/reached``; and the current ``goal/cone_deg`` and
  ``goal/yaw_limit``.
  """

  OUTCOMES = ("reached", "stagnation", "contact", "fell", "timeout")
  TERMS: ClassVar[dict[str, str]] = {
    "stagnation": "stagnation",
    "contact": "illegal_contact",
    "fell": "fell_over",
  }

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.reach_distance = float(cfg.params.get("reach_distance", 0.5))
    self.reach_time_left = float(cfg.params.get("reach_time_left", 4.0))
    dev = self.buffer.outcomes.device
    self.outcome_log = {
      k: OutcomeBuffer(1, 1, GOAL_LOG_EPISODES, dev) for k in self.OUTCOMES
    }
    self.family_log = OutcomeBuffer(self.buffer.families, 1, GOAL_LOG_EPISODES, dev)
    self.distance_log = _FloatRing(GOAL_LOG_EPISODES, dev)
    self._env = env

  def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    floor: int,
    hold_steps: int = 0,
    sgs: dict[str, float] | None = None,
    reach_distance: float = 0.5,
    reach_time_left: float = 4.0,
  ) -> dict[str, Any]:
    del reach_distance, reach_time_left  # resolved at construction
    return super().__call__(
      env, env_ids, command_name, floor, hold_steps=hold_steps, sgs=sgs
    )

  def _judge(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    goal: Any = env.command_manager.get_term(command_name)
    pos = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    dist = torch.norm(goal.pos_w[env_ids, :2] - pos, dim=1)
    t_left = time_left_s(env)[env_ids]
    success = (dist < self.reach_distance) & (t_left < self.reach_time_left)
    zeros = torch.zeros_like(env_ids)
    tm: Any = env.termination_manager
    flags = {"reached": success, "timeout": tm.time_outs[env_ids]}
    for key, term in self.TERMS.items():
      if term in tm.active_terms:
        flags[key] = tm.get_term(term)[env_ids].bool()
      else:
        flags[key] = torch.zeros_like(success)
    for key, value in flags.items():
      self.outcome_log[key].record(zeros, zeros, value)
    terrain: Any = env.scene.terrain
    self.family_log.record(terrain.terrain_types[env_ids], zeros, success)
    self.distance_log.record(dist)
    return success, ~success

  def _sgs_levels(self, terrain: Any, generator: Any) -> dict[str, Any]:
    result = super()._sgs_levels(terrain, generator)
    for key, log in self.outcome_log.items():
      result[f"goal/{key}"] = float(log.success_rate()[0, 0].item())
    result["goal/final_distance"] = self.distance_log.mean()
    names = list(generator.sub_terrains.keys())
    if self.family_log.families == len(names):
      frac = self.family_log.success_rate()[:, 0].tolist()
      for f, name in enumerate(names):
        result[f"{name}/reached"] = frac[f]
    env: Any = self._env
    goal = (
      env.command_manager.get_term(GOAL_COMMAND)
      if hasattr(env, "command_manager")
      else None
    )
    if goal is not None and hasattr(goal, "cone_deg"):
      result["goal/cone_deg"] = goal.cone_deg()
    result["goal/yaw_limit"] = linear_schedule(
      int(getattr(env, "common_step_counter", 0)),
      GOAL_YAW_SCHEDULE["steps"],
      0.0,
      GOAL_YAW_SCHEDULE["limit"],
    )
    return result


__all__ = [
  "GAIT_COMMAND",
  "GOAL_COMMAND",
  "GOAL_LOG_EPISODES",
  "GOAL_OBS_CLIP",
  "GOAL_SCHEDULE_STEPS",
  "GOAL_YAW_SCHEDULE",
  "STAIR_FAMILIES",
  "GoalCommand",
  "GoalCommandCfg",
  "GoalGaitCommand",
  "GoalGaitCommandCfg",
  "goal_distance",
  "goal_heading_tracking",
  "goal_move_to_goal",
  "goal_obs",
  "goal_position_tracking",
  "goal_stagnation",
  "goal_stand_at_goal",
  "lattice_stone_table",
  "linear_schedule",
  "terrain_levels_sgs_goal",
  "time_left",
  "time_left_s",
  "velocity_stagnation",
]
