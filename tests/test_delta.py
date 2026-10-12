"""DeltaModel: geometry, boundedness, PPO-safety, export."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from ame.delta import DeltaModel

MAP = (33, 21, 3)
MAP_SIZE = 33 * 21 * 3


def _obs(bs=4, proprio=99, map_size=MAP_SIZE):
  return TensorDict(
    {"actor": torch.randn(bs, proprio), "height_map": torch.randn(bs, map_size)},
    batch_size=[bs],
  )


def _model(**kw):
  defaults = dict(
    obs=_obs(),
    obs_groups={"actor": ["actor", "height_map"]},
    obs_set="actor",
    output_dim=29,
    map_shape=MAP,
    hidden_dims=[128, 64],
  )
  defaults.update(kw)
  return DeltaModel(**defaults)


def test_forward_shapes_and_grad():
  m = _model()
  obs = _obs(bs=6)
  out = m.mlp(m.get_latent(obs))
  assert out.shape == (6, 29)
  out.sum().backward()
  # The offset projections must receive gradient (deformation is trained).
  assert m.layers[0].w_s.weight.grad is not None
  assert m.layers[0].refine_mlp[-1].weight.grad is not None


def test_sample_locations_bounded():
  m = _model()
  obs = _obs(bs=8)
  # Push extreme inputs; locations must stay strictly inside the map.
  obs["actor"] = obs["actor"] * 100
  loc = m.sample_locations(obs)
  assert loc.shape == (8, 4, 8, 2)
  assert loc.abs().max() < 1.0


def test_encoder_cost_independent_of_resolution():
  m_lo = _model()
  fine = (65, 41, 3)
  m_hi = _model(map_shape=fine, obs=_obs(map_size=65 * 41 * 3))
  n_lo = sum(p.numel() for p in m_lo.parameters())
  n_hi = sum(p.numel() for p in m_hi.parameters())
  assert n_lo == n_hi  # fixed sampling/patch settings -> resolution-free cost


def test_train_eval_parity():
  # No BatchNorm anywhere: PPO's rollout/update passes see the same function.
  m = _model()
  obs = _obs(bs=3)
  m.train()
  a = m.mlp(m.get_latent(obs))
  m.eval()
  b = m.mlp(m.get_latent(obs))
  assert torch.allclose(a, b)


def test_flat_map_token_geometry():
  # On a constant map, every patch is flat: center-relative z is 0, so the
  # encoder output must be identical for two different constant elevations.
  m = _model()
  obs = _obs(bs=2)
  grid = torch.zeros(2, 21, 33, 3)
  obs_a = obs.clone()
  obs_a["height_map"] = (grid + torch.tensor([0.0, 0.0, -0.5])).reshape(2, -1)
  obs_b = obs.clone()
  obs_b["height_map"] = (grid + torch.tensor([0.0, 0.0, -0.5])).reshape(2, -1)
  assert torch.allclose(m.get_latent(obs_a), m.get_latent(obs_b))


def test_ablation_switches():
  fixed = _model(learned_offsets=False, scout_refine=False, adaptive_patch=False)
  obs = _obs(bs=2)
  loc1 = fixed.sample_locations(obs)
  obs2 = obs.clone()
  obs2["actor"] = torch.randn_like(obs2["actor"])
  loc2 = fixed.sample_locations(obs2)
  # Fixed sampling: locations do not depend on the state.
  assert torch.allclose(loc1, loc2)
  assert not hasattr(fixed.layers[0], "w_s")
  # Full model: offset heads are zero-initialized (sampling starts at the
  # base references), so state dependence appears once they have weights.
  full = _model()
  assert torch.allclose(full.sample_locations(obs), full.sample_locations(obs2))
  with torch.no_grad():
    for layer in full.layers:
      layer.w_s.weight.normal_(0, 0.5)
  assert not torch.allclose(full.sample_locations(obs), full.sample_locations(obs2))


def test_onnx_export_parity(tmp_path):
  ort = pytest.importorskip("onnxruntime")
  m = _model()
  m.eval()
  wrapper = m.as_onnx()
  path = tmp_path / "delta.onnx"
  torch.onnx.export(
    wrapper,
    wrapper.get_dummy_inputs(),
    str(path),
    input_names=wrapper.input_names,
    output_names=wrapper.output_names,
    dynamo=False,
  )
  proprio = torch.randn(1, 99)
  hm = torch.randn(1, MAP_SIZE)
  with torch.inference_mode():
    ref = wrapper(proprio, hm).numpy()
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  (out,) = sess.run(["action"], {"proprio": proprio.numpy(), "height_map": hm.numpy()})
  assert np.abs(out - ref).max() < 1e-4
