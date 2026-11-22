"""FootholdV2-Progress task: the two new terms and the task wiring (CPU)."""

from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import asdict

import pytest
import torch

from ame.mdp.foothold_v2 import (
  PAY_FULL,
  PAY_LIFTOFF,
  PAY_NONE,
  PAY_RESET,
  FootholdWindows,
  ProgressBuffer,
  sole_off_fraction,
)

D30_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
V2_TASK = (
  "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1"
)
REPO = pathlib.Path(__file__).resolve().parents[1]
DT = 0.02


# --------------------------------------------------------------------------
# Off fraction.
# --------------------------------------------------------------------------


def test_off_fraction_flat_step_and_miss() -> None:
  z = torch.zeros(1, 2, 15)
  miss = torch.zeros(1, 2, 15, dtype=torch.bool)
  z[0, 1, :5] = -0.051  # 5 points just over 5 cm below the top
  z[0, 1, 5:10] = -0.049  # within delta
  frac = sole_off_fraction(z, miss, 0.05)
  assert frac[0, 0].item() == 0.0
  assert frac[0, 1].item() == pytest.approx(5 / 15)
  miss[0, 0, :3] = True  # a miss is off
  assert sole_off_fraction(z, miss, 0.05)[0, 0].item() == pytest.approx(3 / 15)
  miss[0, 0] = True  # every ray missing: fully off
  assert sole_off_fraction(z, miss, 0.05)[0, 0].item() == 1.0


def test_off_fraction_top_is_per_foot_and_per_step() -> None:
  z = torch.full((1, 2, 15), 0.30)  # left foot on a higher tread
  z[0, 1] = 0.0
  z[0, 1, 0] = -0.2
  frac = sole_off_fraction(z, torch.zeros_like(z, dtype=torch.bool), 0.05)
  assert frac[0, 0].item() == 0.0  # no comparison across feet
  assert frac[0, 1].item() == pytest.approx(1 / 15)


# --------------------------------------------------------------------------
# Touchdown windows.
# --------------------------------------------------------------------------


def _run(seq, window=5):
  """seq: list of (touchdown, in_contact, frac, ending) for one env, one foot."""
  w = FootholdWindows(1, 1, window, "cpu")
  out = []
  for td, c, f, e in seq:
    paid, reason = w.step(
      torch.tensor([[td]]), torch.tensor([[c]]), torch.tensor([[f]]), torch.tensor([e])
    )
    out.append((paid.item(), reason.item()))
  return out


def test_window_full_pays_max_squared_on_fifth_step() -> None:
  # calls 0..4 sample the touchdown step and the next four; pay on call 4.
  seq = [(True, True, 0.2, False)] + [
    (False, True, f, False) for f in (0.6, 0.4, 0.0, 0.1)
  ]
  seq += [(False, True, 1.0, False)]  # after the window: ignored
  out = _run(seq)
  assert [r for _, r in out] == [PAY_NONE] * 4 + [PAY_FULL, PAY_NONE]
  assert out[4][0] == pytest.approx(0.36)
  assert sum(p for p, _ in out) == pytest.approx(0.36)


def test_window_samples_the_touchdown_step() -> None:
  out = _run([(True, True, 0.9, False)] + [(False, True, 0.0, False)] * 4)
  assert out[4] == (pytest.approx(0.81), PAY_FULL)


def test_window_cut_on_liftoff_does_not_sample_liftoff_step() -> None:
  seq = [
    (True, True, 0.0, False),
    (False, True, 0.4, False),
    (False, False, 0.8, False),
  ]
  out = _run(seq)
  assert out[2] == (pytest.approx(0.16), PAY_LIFTOFF)


def test_window_cut_on_episode_end_and_no_start_on_end() -> None:
  out = _run(
    [(True, True, 0.0, False), (False, True, 0.5, False), (False, True, 0.3, True)]
  )
  assert out[2] == (pytest.approx(0.25), PAY_RESET)
  out = _run([(True, True, 0.9, True), (False, True, 0.9, False)])
  assert out == [(0.0, PAY_NONE), (0.0, PAY_NONE)]


def test_window_reset_clears_state() -> None:
  w = FootholdWindows(2, 2, 5, "cpu")
  t = torch.ones(2, 2, dtype=torch.bool)
  w.step(t, t, torch.zeros(2, 2), torch.zeros(2, dtype=torch.bool))
  w.step(~t, t, torch.ones(2, 2), torch.zeros(2, dtype=torch.bool))
  w.reset(torch.tensor([1]))
  assert w.active[0].all() and not w.active[1].any()
  assert w.x[1].eq(0).all() and w.samples[1].eq(0).all()


# --------------------------------------------------------------------------
# Progress buffer.
# --------------------------------------------------------------------------


def _progress(speed_ratio, steps=60, v=(1.0, 0.0), b=1):
  buf = ProgressBuffer(b, 50, "cpu")
  pos = torch.zeros(b, 2)
  out = []
  vel = torch.tensor([v] * b)
  for _ in range(steps):
    pos = pos + vel * speed_ratio * DT
    out.append(buf.step(pos.clone(), vel, DT, 0.8, 0.3))
  return torch.stack(out)  # [steps, b]


def test_progress_needs_full_window() -> None:
  out = _progress(0.0)
  assert out[:50].eq(0).all()  # 50 entries: no full 1 s of displacement yet
  assert out[50:].eq(0.8).all()  # standing still: ratio 0 -> 0.8


def test_progress_ratio_and_clip() -> None:
  assert _progress(1.0)[-1].item() == pytest.approx(0.0, abs=1e-5)
  assert _progress(0.5)[-1].item() == pytest.approx(0.3, abs=1e-5)
  assert _progress(-1.0)[-1].item() == pytest.approx(
    0.8
  )  # backwards: ratio clipped to 0


def test_progress_min_speed_and_reset() -> None:
  assert _progress(0.0, v=(0.25, 0.0)).eq(0).all()  # mean command speed 0.25 <= 0.3
  buf = ProgressBuffer(2, 50, "cpu")
  vel = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
  for _ in range(60):
    buf.step(torch.zeros(2, 2), vel, DT, 0.8, 0.3)
  buf.reset(torch.tensor([0]))
  out = buf.step(torch.zeros(2, 2), vel, DT, 0.8, 0.3)
  assert out[0].item() == 0.0 and out[1].item() == pytest.approx(0.8)


def test_progress_integrates_a_resampled_command() -> None:
  # Half the window commands +x at 1 m/s, half +y at 1 m/s; the robot follows
  # exactly: C = (0.5, 0.5), d = C, ratio 1, no penalty.
  buf = ProgressBuffer(1, 50, "cpu")
  pos = torch.zeros(1, 2)
  for i in range(80):
    v = torch.tensor([[1.0, 0.0]]) if i < 55 else torch.tensor([[0.0, 1.0]])
    pos = pos + v * DT
    out = buf.step(pos.clone(), v, DT, 0.8, 0.3)
  assert out.item() == pytest.approx(0.0, abs=1e-5)
  # The same window, robot only followed the first half: d.C/|C|^2 = 0.5 -> 0.3.
  buf = ProgressBuffer(1, 50, "cpu")
  pos = torch.zeros(1, 2)
  for i in range(80):
    v = torch.tensor([[1.0, 0.0]]) if i < 55 else torch.tensor([[0.0, 1.0]])
    pos = pos + (v * DT if i < 55 else 0.0)
    out = buf.step(pos.clone(), v, DT, 0.8, 0.3)
  # last 50 steps: 25 of +x (followed), 25 of +y (not): d = (0.5, 0), C = (0.5, 0.5)
  assert out.item() == pytest.approx(0.3, abs=1e-5)


# --------------------------------------------------------------------------
# Task wiring.
# --------------------------------------------------------------------------


def _cfgs(task: str, play: bool = False):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import ame  # noqa: F401

  return load_env_cfg(task, play=play), load_rl_cfg(task)


norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731


@pytest.mark.parametrize("play", [False, True])
def test_v2_task_is_d30_with_two_terms_swapped(play: bool) -> None:
  old, _ = _cfgs(D30_TASK, play)
  new, _ = _cfgs(V2_TASK, play)
  assert "foothold" in old.rewards and "foothold" not in new.rewards
  assert new.rewards["foothold_v2"].weight == -1.0
  assert new.rewards["foothold_v2"].params["delta"] == 0.05
  assert new.rewards["progress_shortfall"].weight == -2.0
  for name in old.rewards:
    if name != "foothold":
      assert norm(old.rewards[name]) == norm(new.rewards[name]), name
  old_r, new_r = dict(old.rewards), dict(new.rewards)
  old.rewards, new.rewards = {}, {}
  assert norm(old) == norm(new)  # everything but the rewards identical
  old.rewards, new.rewards = old_r, new_r


def test_v2_task_double_critic_groups() -> None:
  _, old = _cfgs(D30_TASK)
  _, new = _cfgs(V2_TASK)
  assert new.algorithm.reward_group_terms == ("foothold_v2",)
  assert new.algorithm.advantage_weights == (1.0, 0.25)
  a, b = asdict(old), asdict(new)
  a["algorithm"].pop("reward_group_terms")
  b["algorithm"].pop("reward_group_terms")
  assert norm(a) == norm(b)


def test_existing_task_config_hashes_unchanged() -> None:
  """Every task registered before this change keeps its config hashes."""
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  import ame  # noqa: F401

  h = lambda x: hashlib.sha256(norm(x).encode()).hexdigest()[:16]  # noqa: E731
  ref = REPO / "notes" / "foothold_v2_check" / "config_hashes_71cf35f.txt"
  expected = {
    line.split()[0]: line.split()[1:] for line in ref.read_text().splitlines() if line
  }
  assert V2_TASK not in expected
  for task, hashes in expected.items():
    assert task in list_tasks()
    got = [
      h(load_env_cfg(task)),
      h(load_env_cfg(task, play=True)),
      h(asdict(load_rl_cfg(task))),
    ]
    assert got == hashes, task
