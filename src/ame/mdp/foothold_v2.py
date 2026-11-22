"""Foothold v2 and progress shortfall: the two terms of the FootholdV2-Progress task.

Both are defined in notes/feasibility_v2/report.md and measured offline by
notes/edge_measure/measure_v2.py + notes/feasibility_v2/analyze_v2.py; these
are the online versions.

``foothold_v2`` (group 2 of the double critic), paid once per touchdown.

* Touchdown: first contact of a foot (``feet_ground_contact``, ``found > 0``)
  whose preceding swing (``last_air_time``) lasted >= ``min_air`` s, at least
  ``min_time`` s after the reset, on a step that does not end the episode.
* Window: the touchdown step and the next ``window - 1`` policy steps, cut on
  lift-off (contact lost) or on the step that ends the episode.
* A sole point (the 15-point grid of each foot, ``foot_sole_scan``, yaw-aligned
  straight-down rays: terrain under the point's xy, foot tilt ignored) is off
  if its terrain hit z is below (max of the 15 hit z) - ``delta``; a ray miss is
  off (all 15 missing = fraction 1).
* x = max off fraction over the window; the term pays x^2 / dt on the step the
  window closes, so with the manager's x dt the episode reward gets -x^2 per
  touchdown at weight -1, and the manager's ``_step_reward`` mean is the
  per-second value of the feasibility tables.

  Timing (measured, notes/foothold_v2_check/report.md): at reward time the
  sole-scan hits and the contact flag are both this step's (the hits equal the
  ones read after ``env.step`` returns), so the window samples the touchdown
  call and the next ``window - 1`` calls and pays on the last one.  Lift-off
  seen at call s closes the window without sampling step s; on the step that
  ends the episode an open window samples that (pre-reset) step, then pays.
  No window opens on an episode-ending step.
  At reward time the body poses are those of the last physics substep, before
  the step's ``sim.forward()`` (mjlab's own note in ``ManagerBasedRlEnv.step``),
  so a foot landing fast on a stone edge can sit one substep (5 ms) earlier
  than in a post-step log: on stones about 12-17 % of touchdowns read a
  different fraction on the touchdown step (one grid column), on stairs and
  flat ground none do.

``progress_shortfall`` (group 1), every step.

* d = pelvis xy displacement over the last ``window_s`` along the commanded
  planar direction, C = commanded planar displacement over the same window
  (the command's xy velocity rotated to world by the heading at each step,
  times dt, summed: a command resampled inside the window is integrated as it
  was), ratio = max(0, d . C / |C|^2), penalty = max(0, ``target`` - ratio).
* Computed only when the per-env buffer holds a full ``window_s`` since the
  reset and the mean commanded planar speed over the window is > ``min_speed``;
  otherwise 0.  The buffer is cleared on reset.  Per-second convention: the
  term returns the raw penalty, so ``_step_reward`` = weight * penalty.

  Timing: the pose and command at reward time are this step's (the command is
  recomputed after the rewards).  The buffer starts with the first step after
  a reset, so the first value comes 1.0 s + 1 step after the reset (the
  offline analysis used the spawn pose and started one step earlier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg

from ame.mdp.foothold import (
  FEET_CONTACT_SENSOR,
  SOLE_SCAN_SENSOR,
  check_foot_order,
  feet_in_contact,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FOOTHOLD_V2_DELTA = 0.05
"""Off threshold below the highest of the 15 terrain hits, m (feasibility v2)."""
FOOTHOLD_V2_WEIGHT = -1.0
FOOTHOLD_V2_WINDOW = 5
"""Touchdown step + 4: 0.1 s at dt = 0.02."""
FOOTHOLD_V2_MIN_AIR = 0.1
FOOTHOLD_V2_MIN_TIME = 0.5
PROGRESS_WEIGHT = -2.0
PROGRESS_WINDOW_S = 1.0
PROGRESS_TARGET = 0.8
PROGRESS_MIN_SPEED = 0.3
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

PAY_NONE, PAY_FULL, PAY_LIFTOFF, PAY_RESET = 0, 1, 2, 3


def sole_off_fraction(
  hit_z: torch.Tensor, missed: torch.Tensor, delta: float
) -> torch.Tensor:
  """``[..., F]`` share of the 15 points per foot that are off.

  ``hit_z``, ``missed``: ``[..., F, 15]``.  Off = miss, or hit z below the
  highest hit of that foot minus ``delta``; a foot with every ray missing is
  fully off."""
  z = torch.where(missed, torch.full_like(hit_z, -torch.inf), hit_z)
  top = z.max(dim=-1, keepdim=True).values
  off = missed | (z < top - delta)
  return off.float().mean(dim=-1)


class FootholdWindows:
  """Per-env, per-foot touchdown windows (the state machine of ``foothold_v2``).

  ``step`` is called once per policy step with this step's inputs and returns
  the x^2 paid this step ``[B, F]`` and the pay reason ``[B, F]``."""

  def __init__(self, num_envs: int, num_feet: int, window: int, device):
    self.window = window
    self.active = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
    self.samples = torch.zeros(num_envs, num_feet, dtype=torch.long, device=device)
    self.x = torch.zeros(num_envs, num_feet, device=device)

  def step(
    self,
    touchdown: torch.Tensor,  # [B, F] valid touchdown this step
    in_contact: torch.Tensor,  # [B, F] contact flag this step
    frac: torch.Tensor,  # [B, F] off fraction of this step's hits
    ending: torch.Tensor,  # [B] this step ends the episode
  ) -> tuple[torch.Tensor, torch.Tensor]:
    reason = torch.zeros_like(self.samples)
    lift = self.active & ~in_contact  # closes without sampling this step
    stance = self.active & in_contact
    self.x = torch.where(stance, torch.maximum(self.x, frac), self.x)
    self.samples = self.samples + stance.long()
    full = stance & (self.samples >= self.window)
    end = stance & ~full & ending[:, None]
    close = lift | full | end
    reason[full] = PAY_FULL
    reason[lift] = PAY_LIFTOFF
    reason[end] = PAY_RESET
    paid = torch.where(close, self.x**2, torch.zeros_like(self.x))
    self.active = self.active & ~close
    # New windows sample their touchdown step now.
    start = touchdown & ~self.active & ~ending[:, None]
    self.active = self.active | start
    self.samples = torch.where(start, torch.ones_like(self.samples), self.samples)
    self.x = torch.where(
      start, frac, torch.where(close, torch.zeros_like(self.x), self.x)
    )
    return paid, reason

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self.active[env_ids] = False
    self.samples[env_ids] = 0
    self.x[env_ids] = 0.0


class foothold_v2:
  """Reward term: x^2 / dt per closed touchdown window, summed over feet."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.sensor = env.scene[p.get("sensor_name", SOLE_SCAN_SENSOR)]
    self.contact = env.scene[p.get("contact_sensor_name", FEET_CONTACT_SENSOR)]
    check_foot_order(self.contact, self.sensor.cfg)
    robot = env.scene["robot"]
    self.foot_ids = [robot.body_names.index(n) for n in FOOT_BODIES]
    n_feet = len(self.contact.primary_names)
    self.windows = FootholdWindows(
      env.num_envs, n_feet, int(p.get("window", FOOTHOLD_V2_WINDOW)), env.device
    )
    self._num_feet = n_feet
    # Diagnostics, read by notes/foothold_v2_check (not used by training).
    self.paid = torch.zeros(env.num_envs, n_feet, device=env.device)
    self.reason = torch.zeros(env.num_envs, n_feet, dtype=torch.long, device=env.device)
    self.start_xy = torch.zeros(env.num_envs, n_feet, 2, device=env.device)
    self.paid_xy = torch.zeros(env.num_envs, n_feet, 2, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = SOLE_SCAN_SENSOR,
    contact_sensor_name: str = FEET_CONTACT_SENSOR,
    delta: float = FOOTHOLD_V2_DELTA,
    window: int = FOOTHOLD_V2_WINDOW,
    min_air: float = FOOTHOLD_V2_MIN_AIR,
    min_time: float = FOOTHOLD_V2_MIN_TIME,
  ) -> torch.Tensor:
    del sensor_name, contact_sensor_name, window
    dt = env.step_dt
    d = self.sensor.data
    b, f = env.num_envs, self._num_feet
    hit_z = d.hit_pos_w[..., 2].view(b, f, -1)
    missed = (d.distances < 0).view(b, f, -1)
    frac = sole_off_fraction(hit_z, missed, delta)  # this step's hits
    in_contact = feet_in_contact(self.contact)
    first = self.contact.compute_first_contact(dt)
    air = self.contact.data.last_air_time
    assert air is not None
    since = env.episode_length_buf.float() * dt
    touchdown = first & (air >= min_air) & (since >= min_time)[:, None]
    ending = env.reset_buf.bool()
    closing_xy = self.start_xy.clone()
    paid, reason = self.windows.step(touchdown, in_contact, frac, ending)
    feet_xy = env.scene["robot"].data.body_link_pos_w[:, self.foot_ids, :2]
    self.start_xy = torch.where(touchdown[..., None], feet_xy, self.start_xy)
    self.paid, self.reason = paid, reason
    self.paid_xy = torch.where((reason > 0)[..., None], closing_xy, self.paid_xy)
    return paid.sum(dim=-1) / dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.windows.reset(env_ids)


def foothold_v2_reward_cfg(
  delta: float = FOOTHOLD_V2_DELTA, weight: float = FOOTHOLD_V2_WEIGHT
) -> RewardTermCfg:
  return RewardTermCfg(
    func=foothold_v2,
    weight=weight,
    params={
      "sensor_name": SOLE_SCAN_SENSOR,
      "contact_sensor_name": FEET_CONTACT_SENSOR,
      "delta": delta,
      "window": FOOTHOLD_V2_WINDOW,
      "min_air": FOOTHOLD_V2_MIN_AIR,
      "min_time": FOOTHOLD_V2_MIN_TIME,
    },
  )


class ProgressBuffer:
  """Per-env ring buffers of pelvis xy (``n + 1`` entries) and world-frame
  commanded xy velocity (``n`` entries); ``step`` returns the shortfall."""

  def __init__(self, num_envs: int, n: int, device):
    self.n = n
    self.pos = torch.zeros(num_envs, n + 1, 2, device=device)
    self.vel = torch.zeros(num_envs, n, 2, device=device)
    self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

  def step(
    self,
    pos_xy: torch.Tensor,  # [B, 2] pelvis xy this step
    vel_cmd_w: torch.Tensor,  # [B, 2] commanded planar velocity, world frame
    dt: float,
    target: float,
    min_speed: float,
  ) -> torch.Tensor:
    rows = torch.arange(pos_xy.shape[0], device=pos_xy.device)
    self.pos[rows, self.count % (self.n + 1)] = pos_xy
    self.vel[rows, self.count % self.n] = vel_cmd_w
    self.count += 1
    full = self.count >= self.n + 1
    oldest = self.pos[rows, self.count % (self.n + 1)]  # n steps before this one
    disp = pos_xy - oldest
    cmd_disp = self.vel.sum(dim=1) * dt  # [B, 2] over the last n steps
    mean_speed = self.vel.norm(dim=-1).mean(dim=1)
    c2 = (cmd_disp**2).sum(dim=-1)
    ratio = ((disp * cmd_disp).sum(dim=-1) / c2.clamp(min=1e-12)).clamp(min=0.0)
    on = full & (mean_speed > min_speed) & (c2 > 1e-12)
    return torch.where(on, (target - ratio).clamp(min=0.0), torch.zeros_like(ratio))

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self.count[env_ids] = 0
    self.pos[env_ids] = 0.0
    self.vel[env_ids] = 0.0


class progress_shortfall:
  """Reward term: max(0, target - ratio) of the last ``window_s`` (see module)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    n = round(float(cfg.params.get("window_s", PROGRESS_WINDOW_S)) / env.step_dt)
    self.buffer = ProgressBuffer(env.num_envs, n, env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    window_s: float = PROGRESS_WINDOW_S,
    target: float = PROGRESS_TARGET,
    min_speed: float = PROGRESS_MIN_SPEED,
  ) -> torch.Tensor:
    del window_s
    data = env.scene["robot"].data
    cmd = env.command_manager.get_command(command_name)[:, :2]
    h = data.heading_w
    c, s = torch.cos(h), torch.sin(h)
    vel_w = torch.stack(
      [c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], -1
    )
    return self.buffer.step(
      data.root_link_pos_w[:, :2], vel_w, env.step_dt, target, min_speed
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.buffer.reset(env_ids)


def progress_shortfall_reward_cfg(
  weight: float = PROGRESS_WEIGHT, command_name: str = "twist"
) -> RewardTermCfg:
  return RewardTermCfg(
    func=progress_shortfall,
    weight=weight,
    params={
      "command_name": command_name,
      "window_s": PROGRESS_WINDOW_S,
      "target": PROGRESS_TARGET,
      "min_speed": PROGRESS_MIN_SPEED,
    },
  )


__all__ = [
  "FOOTHOLD_V2_DELTA",
  "FOOTHOLD_V2_MIN_AIR",
  "FOOTHOLD_V2_MIN_TIME",
  "FOOTHOLD_V2_WEIGHT",
  "FOOTHOLD_V2_WINDOW",
  "PAY_FULL",
  "PAY_LIFTOFF",
  "PAY_NONE",
  "PAY_RESET",
  "PROGRESS_MIN_SPEED",
  "PROGRESS_TARGET",
  "PROGRESS_WEIGHT",
  "PROGRESS_WINDOW_S",
  "FootholdWindows",
  "ProgressBuffer",
  "foothold_v2",
  "foothold_v2_reward_cfg",
  "progress_shortfall",
  "progress_shortfall_reward_cfg",
  "sole_off_fraction",
]
