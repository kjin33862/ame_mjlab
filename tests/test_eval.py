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


def test_illegal_contact_recorded_names_the_contact_partner() -> None:
  import mujoco

  from ame.eval import CONTACT_PARTNER_ATTR, illegal_contact_recorded

  model = mujoco.MjModel.from_xml_string(
    """<mujoco><worldbody>
      <body name="terrain"><geom name="ground" type="box" size="1 1 .1"/></body>
      <body name="robot/hip"><geom name="hip" size=".1"/></body>
      <body name="robot/wrist"><geom name="wrist" size=".1"/></body>
      <body name="robot/knee"><geom name="knee" size=".1"/></body>
    </worldbody></mujoco>"""
  )
  # geoms: 0 ground, 1 hip, 2 wrist, 3 knee.  Sensor bodies: [hip, knee].
  force = torch.zeros(3, 2, 3)
  force[0, 0, 2] = 5.0  # env 0: hip
  force[1, 1, 2] = 5.0  # env 1: knee
  force[2, 0, 2] = 5.0  # env 2: hip, but no raw contact recorded for it
  contact = SimpleNamespace(
    geom=torch.tensor([[2, 1], [0, 1], [0, 3], [0, 2]]),
    worldid=torch.tensor([0, 0, 1, 2]),
    dist=torch.tensor([-0.02, -0.001, -0.01, -0.01]),
  )
  env = SimpleNamespace(
    scene={
      "undesired_contact": SimpleNamespace(
        data=SimpleNamespace(force=force), primary_names=["hip", "knee"]
      ),
      "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(3, 3))),
    },
    sim=SimpleNamespace(
      mj_model=model, data=SimpleNamespace(nacon=torch.tensor([4]), contact=contact)
    ),
    num_envs=3,
    device="cpu",
  )
  illegal_contact_recorded(env, "undesired_contact", 1.0)
  assert getattr(env, CONTACT_PARTNER_ATTR) == [
    "self:wrist+terrain",  # deepest first
    "terrain",
    "none",
  ]


def test_spawn_asym_adds_one_joint_offset_after_the_joint_reset() -> None:
  from mjlab.envs import mdp as envs_mdp

  from ame.eval import SPAWN_ASYM_JOINT, eval_env_cfg

  assert "spawn_asym" not in eval_env_cfg(num_rows=1, num_cols=1).events
  events = eval_env_cfg(num_rows=1, num_cols=1, spawn_asym=0.1).events
  names = list(events)
  assert names.index("spawn_asym") > names.index("reset_robot_joints")
  term = events["spawn_asym"]
  assert term.mode == "reset" and term.func is envs_mdp.reset_joints_by_offset
  assert term.params["position_range"] == (0.1, 0.1)
  assert term.params["asset_cfg"].joint_names == (SPAWN_ASYM_JOINT,)


def test_deploy_actor_obs_drops_the_velocity_and_is_read_from_the_checkpoint(
  tmp_path: Path,
) -> None:
  from ame.eval import checkpoint_actor_obs, eval_env_cfg

  full = eval_env_cfg(num_rows=1, num_cols=1)
  deploy = eval_env_cfg(num_rows=1, num_cols=1, actor_obs="deploy")
  assert "base_lin_vel" in full.observations["actor"].terms
  assert "base_lin_vel" not in deploy.observations["actor"].terms
  assert "base_lin_vel" in deploy.observations["critic"].terms
  with pytest.raises(ValueError):
    eval_env_cfg(num_rows=1, num_cols=1, actor_obs="none")
  for width, expected in ((99, "full"), (96, "deploy")):
    path = tmp_path / f"model_{width}.pt"
    weight = torch.zeros(64, width)
    torch.save({"actor_state_dict": {"proprio_embedding.weight": weight}}, path)
    assert checkpoint_actor_obs(str(path)) == expected


def test_cli_gait_and_symmetry_flags() -> None:
  ev = _load_evaluate_script()
  parse = lambda *a: ev.build_parser().parse_args(["--random_policy", *a])  # noqa: E731
  args = parse()
  assert args.spawn_asym == 0.0 and not args.stochastic_actions
  assert parse("--spawn_asym").spawn_asym == 0.1
  assert parse("--spawn_asym", "0.2").spawn_asym == 0.2
  assert parse("--stochastic_actions").stochastic_actions
  assert (
    args.actor_obs == "auto" and parse("--actor_obs", "deploy").actor_obs == "deploy"
  )


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


def test_success_counts_only_progress_along_the_spawn_heading() -> None:
  ev = _load_evaluate_script()
  assert ev.build_parser().parse_args(["--random_policy"]).success_rule == "forward"
  # forward 4 m, sideways 4 m, backward 4 m, diagonal (2.83 m forward) at yaw 0; then forward at yaw 90 deg
  disp = torch.tensor([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [2.83, 2.83], [0.0, 4.0]])
  heading = torch.tensor([0.0, 0.0, 0.0, 0.0, torch.pi / 2])
  forward, xy, progress = ev.success_progress(disp, heading, "forward")
  assert torch.allclose(forward, torch.tensor([4.0, 0.0, -4.0, 2.83, 4.0]), atol=1e-5)
  assert torch.equal(progress, forward)
  assert ((progress >= 4.0) == torch.tensor([True, False, False, False, True])).all()
  _, _, any_dir = ev.success_progress(disp, heading, "any")
  assert torch.equal(any_dir, xy) and (any_dir[:3] >= 4.0).all()


@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_stone_lattice_is_the_gap_board_lattice_with_a_stone_spawn(
  difficulty: float,
) -> None:
  import mujoco
  import numpy as np

  from ame.eval import EVAL_TERRAINS, eval_env_cfg
  from ame.eval.terrains import EvalStoneLatticeTerrainCfg
  from ame.terrains import TILE, gap_board_cfg

  assert "stone_lattice" in EVAL_TERRAINS and "stepping_stones" in EVAL_TERRAINS
  cfg = eval_env_cfg(
    terrain="stone_lattice", num_rows=2, num_cols=2, difficulty=difficulty
  )
  gen = cfg.scene.terrain.terrain_generator
  assert list(gen.sub_terrains) == ["stone_lattice"]
  assert gen.difficulty_range == (difficulty, difficulty)
  fam = gen.sub_terrains["stone_lattice"]
  assert isinstance(fam, EvalStoneLatticeTerrainCfg) and fam.flat_patch_sampling is None
  train = gap_board_cfg().sub_terrains["stepping_stones"]
  assert (fam.stone_size, fam.gap_range, fam.size, fam.floor_depth) == (
    train.stone_size,
    train.gap_range,
    train.size,
    train.floor_depth,
  )
  assert fam.gap(difficulty) == pytest.approx(0.45 * difficulty)
  assert fam.size == TILE and fam.stone_size == 0.40

  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  out = fam.function(difficulty, spec, np.random.default_rng(0))
  grid, _ = fam.centres(difficulty)
  # the origin is a stone centre (top at z = 0), the nearest one to the tile centre
  assert out.origin[2] == 0.0
  assert np.isclose(np.linalg.norm(grid - out.origin[:2], axis=1), 0.0).any()
  centre = np.asarray(TILE) / 2
  nearest = np.linalg.norm(grid - centre, axis=1).min()
  assert np.linalg.norm(out.origin[:2] - centre) == pytest.approx(nearest)
  if difficulty == 0.5:  # odd stone count: a stone right at the centre
    assert np.allclose(out.origin[:2], centre)


def test_stepping_stones_eval_terrain_is_unchanged() -> None:
  from ame.eval import eval_env_cfg
  from ame.terrains import ame_terrains_cfg

  gen = eval_env_cfg(terrain="stepping_stones", num_rows=1, num_cols=1).scene.terrain
  fam = gen.terrain_generator.sub_terrains["stepping_stones"]
  ref = ame_terrains_cfg().sub_terrains["stepping_stones"]
  assert type(fam) is type(ref)
  assert {**vars(fam), "proportion": 0} == {**vars(ref), "proportion": 0}


def test_curriculum_audit_matches_the_tracking_rule() -> None:
  from ame.eval import DECISIONS, curriculum_decision

  # far / circled to the time-out fast enough / timed out too slow / fell early / fell late / far but fell early
  distance = torch.tensor([4.5, 1.0, 1.0, 0.5, 0.5, 4.5])
  timed_out = torch.tensor([True, True, True, False, False, False])
  length = torch.tensor([1000, 1000, 1000, 100, 800, 100])
  moved = torch.tensor([700.0, 700.0, 500.0, 50.0, 400.0, 80.0])
  asked = torch.full((6,), 1000.0)
  out = curriculum_decision(
    distance, timed_out, length, 1000, moved, asked, half_tile=4.0
  )
  assert [DECISIONS[i] for i in out.tolist()] == [
    "promote_distance",
    "promote_speed_only",
    "demote",
    "demote",
    "none",
    "promote_distance",
  ]
  # standing for the whole episode asks for nothing: surviving it is "tracked"
  zero = torch.zeros(1)
  out = curriculum_decision(
    zero, torch.tensor([True]), torch.tensor([1000]), 1000, zero, zero, 4.0
  )
  assert DECISIONS[int(out)] == "promote_speed_only"


def test_wrap_angle_and_train_commands_option() -> None:
  from ame.env_cfg import ame_gap_env_cfg
  from ame.eval import eval_env_cfg, wrap_angle

  a = torch.tensor([0.0, 3 * torch.pi / 2, -3 * torch.pi / 2, 7.0])
  assert torch.allclose(
    wrap_angle(a),
    torch.tensor([0.0, -torch.pi / 2, torch.pi / 2, 7.0 - 2 * torch.pi]),
    atol=1e-6,
  )
  fixed = eval_env_cfg(num_rows=1, num_cols=1).commands["twist"]
  assert fixed.ranges.lin_vel_x == (1.0, 1.0) and fixed.rel_standing_envs == 0.0
  train = eval_env_cfg(num_rows=1, num_cols=1, commands="train").commands["twist"]
  gap = ame_gap_env_cfg().commands["twist"]
  assert train.ranges == gap.ranges and not train.heading_command
  assert (
    train.rel_standing_envs,
    train.rel_forward_envs,
    train.resampling_time_range,
  ) == (
    gap.rel_standing_envs,
    gap.rel_forward_envs,
    gap.resampling_time_range,
  )
  with pytest.raises(ValueError):
    eval_env_cfg(num_rows=1, num_cols=1, commands="none")
  ev = _load_evaluate_script()
  args = ev.build_parser().parse_args(["--random_policy"])
  assert (
    args.commands == "fixed" and not args.curriculum_audit and args.escape_dist == 3.0
  )
