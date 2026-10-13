"""Evaluation port: eval env config, noise term, recorded termination, CLI noise
resolution.  No simulator rollouts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _load_evaluate_script():
  spec = importlib.util.spec_from_file_location(
    "evaluate", REPO / "scripts" / "evaluate.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _scan_env(b: int = 5, seed: int = 0):
  """A fake env with the real 33 x 21 scan grid and random terrain heights."""
  from mjlab.sensor import GridPatternCfg

  from ame.scan import MAP_RESOLUTION_M, MAP_SIZE_M

  pattern = GridPatternCfg(size=MAP_SIZE_M, resolution=MAP_RESOLUTION_M)
  r = pattern.generate_rays(mj_model=None, device="cpu")[0].shape[0]
  g = torch.Generator().manual_seed(seed)
  hit = torch.zeros(b, r, 3)
  hit[..., 2] = -0.8 + 0.3 * torch.rand(b, r, generator=g)
  data = SimpleNamespace(
    distances=torch.ones(b, r), hit_pos_w=hit, pos_w=torch.zeros(b, 3)
  )
  sensor = SimpleNamespace(
    data=data, cfg=SimpleNamespace(pattern=pattern, max_distance=5.0)
  )
  return SimpleNamespace(
    scene={"terrain_scan": sensor},
    device="cpu",
    num_envs=b,
    episode_length_buf=torch.ones(b, dtype=torch.long),
  )


def _eval_term(env, **params):
  from ame.eval import elevation_map_eval

  cfg = SimpleNamespace(params={"sensor_name": "terrain_scan", **params})
  return elevation_map_eval(cfg, env)


def test_eval_env_cfg_fixes_board_spawn_and_command() -> None:
  from ame.eval import eval_env_cfg, illegal_contact_recorded

  cfg = eval_env_cfg(num_rows=3, num_cols=4, difficulty=0.25, seed=7, episode_s=20.0)
  gen = cfg.scene.terrain.terrain_generator
  assert list(gen.sub_terrains) == ["stepping_stones"]
  assert (gen.num_rows, gen.num_cols, gen.curriculum, gen.seed) == (3, 4, False, 7)
  assert gen.difficulty_range == (0.25, 0.25)
  assert cfg.scene.num_envs == 12 and cfg.seed == 7 and cfg.episode_length_s == 20.0
  assert "randomize_terrain" not in cfg.events and "base_com" not in cfg.events
  assert "foot_friction" in cfg.events  # friction DR stays, as in the Isaac Lab tool
  pose = cfg.events["reset_base"].params["pose_range"]
  assert pose["x"] == pose["y"] == pose["yaw"] == (0.0, 0.0)
  twist = cfg.commands["twist"]
  assert (twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z) == (
    (1.0, 1.0),
    (0.0, 0.0),
    (0.0, 0.0),
  )
  assert twist.rel_standing_envs == 0.0 and not twist.heading_command
  assert cfg.terminations["illegal_contact"].func is illegal_contact_recorded
  assert set(cfg.terminations) == {
    "illegal_contact",
    "bad_orientation",
    "physics_blowup",
    "time_out",
  }


def test_eval_noise_presets_select_the_right_term() -> None:
  from ame.eval import (
    OFF_HEIGHT_NOISE,
    TRAIN_HEIGHT_NOISE,
    elevation_map_eval,
    eval_env_cfg,
  )
  from ame.mdp.observations import elevation_map

  train = eval_env_cfg(height_noise=dict(TRAIN_HEIGHT_NOISE), proprio_noise=True)
  term = train.observations["height_map"].terms["elevation"]
  assert term.func is elevation_map and term.params["noise"] is True
  assert train.observations["actor"].enable_corruption is True

  off = eval_env_cfg(height_noise=dict(OFF_HEIGHT_NOISE), proprio_noise=False)
  term = off.observations["height_map"].terms["elevation"]
  assert term.func is elevation_map and term.params["noise"] is False
  assert off.observations["actor"].enable_corruption is False

  custom = eval_env_cfg(height_noise={**TRAIN_HEIGHT_NOISE, "drift_cells": 2})
  term = custom.observations["height_map"].terms["elevation"]
  assert term.func is elevation_map_eval and term.params["drift_cells"] == 2


def test_training_cfg_is_untouched_by_eval_cfg() -> None:
  from ame.env_cfg import ame_rough_env_cfg
  from ame.eval import eval_env_cfg
  from ame.mdp.terminations import illegal_contact

  eval_env_cfg()
  cfg = ame_rough_env_cfg()
  assert cfg.terminations["illegal_contact"].func is illegal_contact
  assert len(cfg.scene.terrain.terrain_generator.sub_terrains) == 12
  assert cfg.observations["height_map"].terms["elevation"].params["noise"] is True


def test_eval_map_without_noise_equals_the_clean_training_map() -> None:
  from ame.mdp.observations import elevation_map

  env = _scan_env()
  term = _eval_term(
    env, height_sigma=0.0, height_offset=0.0, drift_cells=0, delay_steps=1
  )
  ref = elevation_map(env, "terrain_scan", noise=False)
  assert torch.equal(term(env, "terrain_scan", 0.0, 0.0, 0, 1), ref)


@pytest.mark.parametrize("dx", [-1, 0, 1])
@pytest.mark.parametrize("dy", [-1, 0, 1])
def test_eval_map_reproduces_training_noise_bit_exactly(dx: int, dy: int) -> None:
  """Same drift, offset and RNG state -> the same map as the training term."""
  from ame.mdp import observations as obs

  env = _scan_env()
  b = env.num_envs
  term = _eval_term(
    env, height_sigma=0.03, height_offset=0.05, drift_cells=1, delay_steps=1
  )
  term._dx[:] = dx
  term._dy[:] = dy
  term._offset[:] = torch.linspace(-0.05, 0.05, b).unsqueeze(1)
  # The training term keeps its per-env constants on the env.
  setattr(env, obs._OFFSET_KEY, term._offset.clone())
  setattr(
    env, obs._DRIFT_KEY, torch.full((b,), (dy + 1) * 3 + (dx + 1), dtype=torch.long)
  )

  torch.manual_seed(3)
  ref = obs.elevation_map(env, "terrain_scan", noise=True)
  torch.manual_seed(3)
  out = term(env, "terrain_scan", 0.03, 0.05, 1, 1)
  assert torch.equal(out, ref)


def test_eval_map_drift_offset_and_delay() -> None:
  env = _scan_env(b=2)
  z_of = lambda m: m.reshape(2, 21, 33, 3)[..., 2]  # noqa: E731  (iy, ix) layout
  clean = z_of(
    _eval_term(env, height_sigma=0.0, height_offset=0.0, drift_cells=0, delay_steps=1)(
      env, "terrain_scan", 0.0, 0.0, 0, 1
    )
  )

  term = _eval_term(
    env, height_sigma=0.0, height_offset=0.0, drift_cells=2, delay_steps=1
  )
  term._dx[:] = torch.tensor([2, -2])
  term._dy[:] = torch.tensor([0, 1])
  z = z_of(term(env, "terrain_scan", 0.0, 0.0, 2, 1))
  assert torch.equal(z[0, :, :-2], clean[0, :, 2:])  # content moved by two cells in x
  assert torch.equal(z[0, :, -1], clean[0, :, -1])  # edge clamped
  assert torch.equal(z[1, :-1, 2:], clean[1, 1:, :-2])

  term = _eval_term(
    env, height_sigma=0.0, height_offset=0.1, drift_cells=0, delay_steps=1
  )
  assert term._offset.abs().max() <= 0.1
  z = z_of(term(env, "terrain_scan", 0.0, 0.1, 0, 1))
  assert torch.allclose(z, (clean + term._offset[:, :, None]).clamp(-1.2, 0.0))

  term = _eval_term(
    env, height_sigma=0.0, height_offset=0.0, drift_cells=0, delay_steps=5
  )
  env.episode_length_buf[:] = 0
  first = term(env, "terrain_scan", 0.0, 0.0, 0, 5)
  env.scene["terrain_scan"].data.hit_pos_w[..., 2] -= 0.1
  env.episode_length_buf[:] = torch.tensor([3, 5])
  held = term(env, "terrain_scan", 0.0, 0.0, 0, 5)
  assert torch.equal(held[0], first[0])  # step 3: still the old map
  assert not torch.equal(held[1], first[1])  # step 5: refreshed


def test_illegal_contact_recorded_matches_and_records_the_body() -> None:
  from ame.eval import CONTACT_BODY_ATTR, PRE_RESET_POS_ATTR, illegal_contact_recorded
  from ame.mdp.terminations import illegal_contact

  force = torch.zeros(3, 4, 3)
  force[0, 2, 2] = 5.0  # env 0: body 2 above threshold
  force[1, 1, 0] = 0.5  # env 1: below threshold
  force[2, 0, 1], force[2, 3, 1] = 2.0, 9.0  # env 2: two bodies, body 3 strongest
  pos = torch.arange(9.0).reshape(3, 3)
  env = SimpleNamespace(
    scene={
      "undesired_contact": SimpleNamespace(data=SimpleNamespace(force=force)),
      "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos)),
    },
    num_envs=3,
    device="cpu",
  )
  got = illegal_contact_recorded(env, "undesired_contact", 1.0)
  assert torch.equal(got, illegal_contact(env, "undesired_contact", 1.0))
  assert getattr(env, CONTACT_BODY_ATTR).tolist() == [2, -1, 3]
  assert torch.equal(getattr(env, PRE_RESET_POS_ATTR), pos)


def test_assign_tiles_gives_one_robot_per_tile() -> None:
  from ame.eval import assign_tiles

  rows, cols, n = 3, 4, 12
  origins = torch.arange(rows * cols * 3, dtype=torch.float).reshape(rows, cols, 3)
  terrain = SimpleNamespace(
    terrain_origins=origins,
    terrain_levels=torch.zeros(n, dtype=torch.long),
    terrain_types=torch.zeros(n, dtype=torch.long),
    env_origins=torch.zeros(n, 3),
  )
  env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), num_envs=n)
  assign_tiles(env)
  assert terrain.terrain_levels.tolist() == [i // cols for i in range(n)]
  assert terrain.terrain_types.tolist() == [i % cols for i in range(n)]
  assert torch.equal(terrain.env_origins, origins.reshape(-1, 3))


def test_cli_noise_resolution_and_tags() -> None:
  ev = _load_evaluate_script()
  parse = lambda *a: ev.build_parser().parse_args(["--random_policy", *a])  # noqa: E731

  h, p = ev.resolve_noise(parse())
  assert (
    h
    == {"height_sigma": 0.03, "height_offset": 0.05, "drift_cells": 1, "delay_steps": 1}
    and p
  )
  assert ev.noise_tag(h, p) == "h0.03-o0.05-dc1-d1-p1"

  h, p = ev.resolve_noise(parse("--noise", "off"))
  assert ev.noise_tag(h, p) == "noiseoff"

  h, p = ev.resolve_noise(
    parse("--height_sigma", "0", "--height_offset", "0", "--drift_cells", "0")
  )
  assert ev.noise_tag(h, p) == "h0-o0-dc0-d1-p1"  # proprio only

  h, p = ev.resolve_noise(
    parse("--noise", "off", "--map_delay_steps", "5", "--proprio_noise", "on")
  )
  assert ev.noise_tag(h, p) == "h0-o0-dc0-d5-p1"
