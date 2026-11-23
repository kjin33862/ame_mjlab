"""AMP component and the two AMP tasks' wiring (CPU)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ame  # noqa: F401
from ame.amp import (
  AMP_BODY_NAMES,
  AMP_CLIPS,
  AmpDoubleCriticPPO,
  AmpV2DoubleCriticPPO,
  AmpNormalizer,
  Discriminator,
  ReplayBuffer,
  amp_observation,
)
from ame.mdp.terminations import ame2_torso_collision

V2_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1"
D30_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
AMP1 = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-AMP-Velocity-Rough-G1"
AMPV2 = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-AMPv2-Velocity-Rough-G1"
AMP2 = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-AME2Contact-FootholdD30-AMP-Velocity-Rough-G1"


def test_clip_list():
  assert len(AMP_CLIPS) == len(set(AMP_CLIPS)) == 17
  walk = [c for c in AMP_CLIPS if "walk" in c]
  jog = [c for c in AMP_CLIPS if "jog" in c]
  idle = [c for c in AMP_CLIPS if c.startswith(("idle_turn", "step_rotate"))]
  assert (len(walk), len(jog), len(idle)) == (8, 6, 3)
  assert not any("fall" in c or "getup" in c.lower() for c in AMP_CLIPS)


def test_observation_layout():
  n, k = 5, len(AMP_BODY_NAMES)
  q = torch.nn.functional.normalize(torch.randn(n, k, 4), dim=-1)
  pos, lin, ang = torch.randn(n, k, 3), torch.randn(n, k, 3), torch.randn(n, k, 3)
  # anchor = identity at the origin: positions pass through, velocities rotate into the body frame
  obs = amp_observation(pos, q, lin, ang, torch.zeros(n, 3), torch.tensor([1.0, 0, 0, 0]).expand(n, 4))
  assert obs.shape == (n, 15 * k) == (n, 195)
  assert torch.allclose(obs[:, : 3 * k], pos.reshape(n, -1), atol=1e-6)
  # identity body rotation -> ori block = first two columns of I
  q_id = torch.tensor([1.0, 0, 0, 0]).expand(n, k, 4)
  obs = amp_observation(pos, q_id, lin, ang, torch.zeros(n, 3), q_id[:, 0])
  assert torch.allclose(obs[:, 3 * k : 9 * k], torch.tensor([1.0, 0, 0, 1, 0, 0]).repeat(n, k))
  assert torch.allclose(obs[:, 9 * k : 12 * k], lin.reshape(n, -1))
  assert torch.allclose(obs[:, 12 * k :], ang.reshape(n, -1))


def test_normalizer_matches_running_mean_std():
  norm = AmpNormalizer(3)
  mean, var, count = np.zeros(3), np.ones(3), 1e-4
  for _ in range(4):
    x = np.random.randn(50, 3) * 2 + 1
    norm.update(torch.tensor(x, dtype=torch.float32))
    bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
    delta, tot = bm - mean, count + bc
    var = (var * count + bv * bc + delta**2 * count * bc / tot) / tot
    mean, count = mean + delta * bc / tot, tot
  assert np.allclose(norm.mean.numpy(), mean, atol=1e-5)
  assert np.allclose(norm.var.numpy(), var, atol=1e-5)
  y = torch.tensor([[100.0, 1.0, 1.0]])
  out = norm.normalize(y)
  assert out[0, 0].item() == 10.0  # clipped
  assert torch.allclose(out[0, 1:], torch.tensor((1.0 - mean[1:]) / np.sqrt(var[1:] + 1e-4), dtype=torch.float32), atol=1e-5)


def test_replay_buffer_wraps():
  buf = ReplayBuffer(2, 5, "cpu")
  buf.insert(torch.arange(8.0).reshape(4, 2), -torch.arange(8.0).reshape(4, 2))
  buf.insert(torch.full((3, 2), 9.0), torch.full((3, 2), -9.0))
  assert buf.num_samples == 5 and buf.step == 2
  assert buf.states[:2].eq(9).all() and buf.states[4].eq(9).all()
  assert torch.equal(buf.states[2], torch.tensor([4.0, 5.0]))
  s, sn = next(buf.feed_forward_generator(1, 64))
  assert torch.equal(s, -sn)


def test_grad_pen_formula():
  disc = Discriminator(4, (8,))
  s, sn = torch.randn(6, 2), torch.randn(6, 2)
  x = torch.cat([s, sn], -1).requires_grad_(True)
  (g,) = torch.autograd.grad(disc(x).sum(), x)
  assert torch.allclose(disc.grad_pen(s, sn, 10.0), 10.0 * g.norm(dim=1).pow(2).mean())


def _stub_alg(style):
  alg = AmpDoubleCriticPPO.__new__(AmpDoubleCriticPPO)
  alg.amp_task_reward_lerp = 0.75
  alg.amp_reward_coef = 0.1
  alg._style = style
  alg.reward_groups = lambda r: torch.stack([r - 1.0, torch.ones_like(r)], dim=-1)
  return alg


def test_style_mixed_into_group_one_only():
  alg = _stub_alg(torch.tensor([0.1, 0.0]))
  g = alg.group_rewards(torch.tensor([3.0, 5.0]))
  # group 1 = 0.25 * style + 0.75 * task group 1; group 2 untouched
  assert torch.allclose(g, torch.tensor([[0.25 * 0.1 + 0.75 * 2.0, 1.0], [0.75 * 4.0, 1.0]]))


def test_style_reward_formula():
  alg = _stub_alg(None)
  alg.discriminator = SimpleNamespace()
  d = torch.tensor([[1.0], [0.0], [-1.0], [3.5]])
  alg.discriminator = lambda x: d
  alg.amp_normalizer = SimpleNamespace(normalize=lambda x: x)
  r, dd = alg.style_reward(torch.zeros(4, 1), torch.zeros(4, 1))
  assert torch.allclose(r, 0.1 * torch.tensor([1.0, 0.75, 0.0, 0.0]))
  assert torch.equal(dd, d.squeeze(-1))


def test_torso_collision_rule():
  hist = torch.zeros(3, 2, 4, 3)
  hist[0, 0, :, 2] = 401.0  # torso above 400 N over the whole step
  hist[1, 1, 0, 2] = 1000.0  # pelvis spike in one substep: mean 250 N
  hist[2, 0, :, 2] = 500.0  # above, but before start_time
  env = SimpleNamespace(
    scene={"torso_contact": SimpleNamespace(data=SimpleNamespace(force_history=hist))},
    episode_length_buf=torch.tensor([60, 60, 10]),
    step_dt=0.02,
  )
  out = ame2_torso_collision(env, "torso_contact", 400.0, 1.0)
  assert out.tolist() == [True, False, False]


def _ser(o):
  """Config -> comparable structure (functions by qualified name, not address)."""
  if dataclasses.is_dataclass(o) and not isinstance(o, type):
    return (type(o).__qualname__, {f.name: _ser(getattr(o, f.name)) for f in dataclasses.fields(o)})
  if isinstance(o, dict):
    return {k: _ser(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [_ser(v) for v in o]
  if callable(o):
    return f"{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', type(o).__qualname__)}"
  return o


def _plain(cfg):
  return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg


def test_task1_is_footholdv2_plus_amp():
  assert _ser(load_env_cfg(AMP1)) == _ser(load_env_cfg(V2_TASK))
  a, b = _plain(load_rl_cfg(AMP1)), _plain(load_rl_cfg(V2_TASK))
  alg_a, alg_b = a.pop("algorithm"), b.pop("algorithm")
  assert a == b
  assert alg_a["class_name"] == "ame.amp:AmpDoubleCriticPPO"
  amp_keys = {k for k in alg_a if k.startswith("amp_")}
  assert {k: v for k, v in alg_a.items() if k not in amp_keys | {"class_name"}} == {
    k: v for k, v in alg_b.items() if k != "class_name"
  }
  assert alg_a["reward_group_terms"] == ("foothold_v2",)
  assert tuple(alg_a["advantage_weights"]) == (1.0, 0.25)
  assert (alg_a["amp_reward_coef"], alg_a["amp_task_reward_lerp"]) == (0.1, 0.75)


def test_task2_contact_rule():
  base, cfg = load_env_cfg(D30_TASK), load_env_cfg(AMP2)
  assert "illegal_contact" in base.terminations and "illegal_contact" not in cfg.terminations
  assert set(cfg.terminations) == set(base.terminations) - {"illegal_contact"} | {"torso_collision"}
  t = cfg.terminations["torso_collision"]
  assert t.params["threshold"] == 400.0 and t.params["start_time"] == 1.0 and not t.time_out
  assert cfg.rewards["undesired_contacts"].weight == -1.0
  sensors = {s.name: s for s in cfg.scene.sensors}
  assert sensors["torso_contact"].history_length == cfg.decimation == 4
  assert set(sensors) - {s.name for s in base.scene.sensors} == {"torso_contact"}
  assert _ser(cfg.rewards) == _ser(base.rewards)
  alg = _plain(load_rl_cfg(AMP2))["algorithm"]
  assert alg["reward_group_terms"] == ("foothold",)


def test_ampv2_task_is_d30_plus_ampv2():
  assert _ser(load_env_cfg(AMPV2)) == _ser(load_env_cfg(D30_TASK))
  a, b = _plain(load_rl_cfg(AMPV2)), _plain(load_rl_cfg(D30_TASK))
  alg_a, alg_b = a.pop("algorithm"), b.pop("algorithm")
  assert a == b
  assert alg_a["class_name"] == "ame.amp:AmpV2DoubleCriticPPO"
  amp_keys = {k for k in alg_a if k.startswith("amp_")}
  assert {k: v for k, v in alg_a.items() if k not in amp_keys | {"class_name"}} == {
    k: v for k, v in alg_b.items() if k != "class_name"
  }
  assert alg_a["reward_group_terms"] == ("foothold",)
  assert tuple(alg_a["advantage_weights"]) == (1.0, 0.25)
  assert alg_a["amp_task_reward_lerp"] == 0.5 and alg_a["amp_reward_coef"] == 0.1
  assert (alg_a["amp_disc_learning_rate"], alg_a["amp_disc_num_learning_epochs"], alg_a["amp_disc_num_mini_batches"]) == (1e-3, 5, 4)
  assert alg_a["amp_replay_buffer_size"] == 100_000 and alg_a["amp_grad_pen_lambda"] == 10.0


def test_ampv2_discriminator_schedule():
  alg = AmpV2DoubleCriticPPO.__new__(AmpV2DoubleCriticPPO)
  alg.amp_disc_learning_rate, alg.amp_disc_num_learning_epochs, alg.amp_disc_num_mini_batches = 1e-3, 5, 4
  alg.amp_grad_pen_lambda = 10.0
  dim = 3
  alg.discriminator = Discriminator(2 * dim, (8,))
  alg.amp_normalizer = AmpNormalizer(dim)
  alg._attach_discriminator_optimizer()
  groups = alg.disc_optimizer.param_groups
  assert [g["weight_decay"] for g in groups] == [1e-3, 1e-1] and all(g["lr"] == 1e-3 for g in groups)
  alg.storage = SimpleNamespace(num_envs=8, num_transitions_per_env=24)
  alg.amp_buffer = ReplayBuffer(dim, 100, "cpu")
  alg.amp_buffer.insert(torch.randn(100, dim), torch.randn(100, dim))
  sizes = []

  class Expert:
    def feed_forward_generator(self, n, mbs):
      for _ in range(n):
        sizes.append(mbs)
        yield torch.randn(mbs, dim) + 3, torch.randn(mbs, dim) + 3

  alg.motion = Expert()
  w0 = alg.discriminator.amp_linear.weight.clone()
  out = alg.update_discriminator()
  assert alg.disc_schedule() == (20, 8 * 24 // 4)
  assert sizes == [48] * 20 and out["amp_disc_updates"] == 20.0 and out["amp_disc_mini_batch_size"] == 48.0
  assert out["amp_disc_lr"] == 1e-3 and groups[0]["lr"] == 1e-3  # fixed: no scheduler touches it
  assert not torch.equal(w0, alg.discriminator.amp_linear.weight)
  assert alg.amp_normalizer.count.item() > 20 * 2 * 48 - 1


if __name__ == "__main__":
  pytest.main([__file__, "-q"])
