"""DELTA: deformable elevation-based local terrain attention (arXiv:2608.22033).

Port of Park, Jung & Hwangbo's DELTA encoder onto the same rsl-rl model API
and observation contract as :class:`ame.models.AmeModel` — the flat
``height_map`` group is identical, so DELTA and AME checkpoints train on the
same environment and are directly comparable.

Instead of AME's dense CNN over every map cell, each attention head encodes
only ``num_points`` sampled locations (paper defaults: 3 layers x 4 heads x 8
points = 32 tokens vs AME's 693 cells), through four stages per layer:

1. **State-conditioned sampling** — fixed base-grid references plus a bounded
   query-conditioned offset ``gamma * tanh(W_s q)``; coverage is preserved by
   the references, deformation is local by the tanh bound.
2. **Scout-guided refinement** — a small bilinear patch at the initial
   location is encoded and used to predict a second offset (coarse-to-fine).
3. **Adaptive patch encoding** — the final layer samples a wide patch split
   into center + context with a learned sigmoid gate ``phi = c + eta * P(u)``
   so surrounding context enters only as needed (the paper's generalization
   driver: 56 -> 97 % on unseen mixed courses).
4. **Single-query multi-head cross-attention** with value-only non-affine
   RMSNorm, residual + LN + FFN query update.

Encoder cost is fixed w.r.t. map resolution.  All norms are LN/RMS — no
BatchNorm — so train/eval forward passes are identical (PPO-safe by
construction).

Ablation switches mirror the paper's Table VI and default to the full model:
``learned_offsets=False`` freezes sampling to the base grid (their "Fixed
Sampling"), ``scout_refine=False`` drops stage 2, ``adaptive_patch=False``
replaces stage 3 with the plain scout-patch token.

Port-level choices the paper leaves open, documented here: base references
are a heads-by-points grid (each head owns one lateral band spanning the full
map length) with fixed +-0.05 perturbations; token coordinates ``xi`` are the
normalized sample location plus the center elevation mapped from our
``[-1.2, 0]`` clamp onto ``[-1, 1]``; per-layer parameters are independent
(the scout/patch encoders are shared across heads and points within a layer).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from ame.models import _OnnxAmeModel

_Z_CLIP = (-1.2, 0.0)

# Actor/critic encoder sharing (AME tuning guide: "We share the encoder for
# actor and critic ... this helps speed up convergence, given that the critic
# loss has better supervision signals").  The actor builds the encoder and
# registers it here; the critic adopts the *same module objects* but binds
# them with object.__setattr__ so they are NOT re-registered as its own
# submodules.  That matters: rsl-rl builds its optimizer from
# chain(actor.parameters(), critic.parameters()), and a doubly-listed
# parameter would be stepped twice per update.  Autograd still routes the
# critic's loss into the shared weights, which is the whole point.
_SHARED: dict[str, dict] = {}


def _share_key(**geom) -> str:
  return repr(sorted(geom.items()))


def _mlp(dims: list[int]) -> nn.Sequential:
  layers: list[nn.Module] = []
  for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2:
      layers.append(nn.ELU())
  return nn.Sequential(*layers)


def _norm_z(z: torch.Tensor) -> torch.Tensor:
  """Map the elevation clamp range [-1.2, 0] onto [-1, 1]."""
  lo, hi = _Z_CLIP
  return (z - lo) / (hi - lo) * 2.0 - 1.0


class _DeformableTerrainLayer(nn.Module):
  """One deformable terrain attention layer (paper Fig. 2b)."""

  def __init__(
    self,
    D: int,
    num_heads: int,
    num_points: int,
    scout_patch: int,
    final_patch: int,
    final_center: int,
    offset_scale: float,
    ffn_hidden: int,
    enc_hidden: int,
    is_final: bool,
    learned_offsets: bool,
    scout_refine: bool,
    adaptive_patch: bool,
  ) -> None:
    super().__init__()
    self.H, self.K, self.D = num_heads, num_points, D
    self.d_h = D // num_heads
    self.gamma = offset_scale
    self.scout_patch = scout_patch
    self.final_patch = final_patch
    self.final_center = final_center
    self.is_final = is_final
    self.learned_offsets = learned_offsets
    self.scout_refine = scout_refine
    self.adaptive_patch = adaptive_patch and is_final
    feat = D - 3  # token = [phi (feat); xi (3)]

    if learned_offsets:
      self.w_s = nn.Linear(D, num_heads * num_points * 2)
      nn.init.zeros_(self.w_s.weight)
      nn.init.zeros_(self.w_s.bias)
    self.scout_mlp = _mlp([scout_patch**2, enc_hidden, feat])
    if scout_refine and learned_offsets:
      self.refine_mlp = _mlp([D + D, enc_hidden, 2])
      final = cast(nn.Linear, self.refine_mlp[-1])
      nn.init.zeros_(final.weight)
      nn.init.zeros_(final.bias)
    if self.adaptive_patch:
      self.f_center = _mlp([final_center**2, enc_hidden, feat])
      self.f_ctx = _mlp([final_patch**2, enc_hidden, feat])
      self.f_gate = nn.Linear(2 * feat, 1)
      self.p_ctx = nn.Linear(feat, feat)

    self.w_k = nn.Parameter(torch.empty(num_heads, D, self.d_h))
    self.w_v = nn.Parameter(torch.empty(num_heads, D, self.d_h))
    nn.init.xavier_uniform_(self.w_k)
    nn.init.xavier_uniform_(self.w_v)
    self.w_o = nn.Linear(D, D)
    self.ln_attn = nn.LayerNorm(D)
    self.ln_ffn = nn.LayerNorm(D)
    self.ffn = _mlp([D, ffn_hidden, D])

  # -- geometry helpers -------------------------------------------------------

  def _bounds(self, patch: int, cell: torch.Tensor) -> torch.Tensor:
    """Per-axis valid patch-center region: patch stays inside the map."""
    half = (patch - 1) / 2
    return (1.0 - half * cell).clamp_min(0.05)

  @staticmethod
  def _bound(r: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    return ell * torch.tanh(r / ell)

  @staticmethod
  def _sample_patch(
    zimg: torch.Tensor, centers: torch.Tensor, patch: int, cell: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear P x P patches (center-relative) + center z.

    zimg: [B, 1, W, L]; centers: [B, N, 2] normalized (x, y).
    Returns (rel_patch [B, N, P*P], center_z [B, N]).
    """
    b, n, _ = centers.shape
    half = (patch - 1) / 2
    steps = torch.arange(patch, device=centers.device, dtype=centers.dtype) - half
    ox, oy = torch.meshgrid(steps * cell[0], steps * cell[1], indexing="xy")
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [P*P, 2]
    grid = centers.unsqueeze(2) + offs.view(1, 1, -1, 2)  # [B, N, P*P, 2]
    vals = F.grid_sample(
      zimg, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[:, 0]  # [B, N, P*P]
    center = F.grid_sample(
      zimg, centers.view(b, n, 1, 2), mode="bilinear", align_corners=True
    )[:, 0, :, 0]  # [B, N]
    return vals - center.unsqueeze(-1), center

  def _token(
    self, zimg: torch.Tensor, loc: torch.Tensor, cell: torch.Tensor
  ) -> torch.Tensor:
    """Terrain evidence token [B, N, D] at the given locations."""
    if self.adaptive_patch:
      rel_c, center = self._sample_patch(zimg, loc, self.final_center, cell)
      rel_u, _ = self._sample_patch(zimg, loc, self.final_patch, cell)
      c = self.f_center(rel_c)
      u = self.f_ctx(rel_u)
      eta = torch.sigmoid(self.f_gate(torch.cat([c, u], dim=-1)))
      phi = c + eta * self.p_ctx(u)
    else:
      rel, center = self._sample_patch(zimg, loc, self.scout_patch, cell)
      phi = self.scout_mlp(rel)
    xi = torch.cat([loc, _norm_z(center).unsqueeze(-1)], dim=-1)
    return torch.cat([phi, xi], dim=-1)

  # -- forward ----------------------------------------------------------------

  def forward(
    self, q: torch.Tensor, refs: torch.Tensor, zimg: torch.Tensor, cell: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer.  q: [B, D]; refs: [B, H, K, 2].  Returns (q_next, refined refs)."""
    b = q.shape[0]
    ell_scout = self._bounds(self.scout_patch, cell)
    if self.learned_offsets:
      d0 = self.gamma * torch.tanh(self.w_s(q)).view(b, self.H, self.K, 2)
      r_tilde = refs + d0
    else:
      r_tilde = refs
    s0 = self._bound(r_tilde, ell_scout)

    if self.scout_refine and self.learned_offsets:
      g = self._scout_token(zimg, s0, cell)  # [B, H, K, D]
      q_b = q.view(b, 1, 1, self.D).expand(b, self.H, self.K, self.D)
      d1 = self.gamma * torch.tanh(self.refine_mlp(torch.cat([q_b, g], dim=-1)))
      r_next = r_tilde + d1
    else:
      r_next = r_tilde
    patch = self.final_patch if self.adaptive_patch else self.scout_patch
    loc = self._bound(r_next, self._bounds(patch, cell))

    tokens = self._token(zimg, loc.view(b, -1, 2), cell).view(b, self.H, self.K, self.D)
    kk = torch.einsum("bhkd,hde->bhke", tokens, self.w_k)
    vv = torch.einsum("bhkd,hde->bhke", tokens, self.w_v)
    vv = vv * torch.rsqrt(vv.pow(2).mean(-1, keepdim=True) + 1e-8)
    qh = q.view(b, self.H, self.d_h)
    attn = torch.softmax(torch.einsum("bhe,bhke->bhk", qh, kk) / self.d_h**0.5, dim=-1)
    ctx = torch.einsum("bhk,bhke->bhe", attn, vv).reshape(b, self.D)
    q_hat = self.ln_attn(q + self.w_o(ctx))
    q_next = self.ln_ffn(q_hat + self.ffn(q_hat))
    return q_next, r_next

  def _scout_token(
    self, zimg: torch.Tensor, loc: torch.Tensor, cell: torch.Tensor
  ) -> torch.Tensor:
    b = loc.shape[0]
    rel, center = self._sample_patch(zimg, loc.view(b, -1, 2), self.scout_patch, cell)
    feat = self.scout_mlp(rel)
    xi = torch.cat([loc.view(b, -1, 2), _norm_z(center).unsqueeze(-1)], dim=-1)
    return torch.cat([feat, xi], dim=-1).view(b, self.H, self.K, self.D)


class DeltaModel(MLPModel):
  """MLP actor/critic with the DELTA deformable terrain encoder in front.

  Same observation contract and framing as :class:`ame.models.AmeModel`
  (the map is the ``map_group`` observation group; the latent fed to the MLP
  is terrain representation ⊕ raw proprio), so the two are drop-in
  alternatives selected by ``class_name`` in the RL config.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    map_group: str = "height_map",
    map_shape: tuple[int, int, int] = (33, 21, 3),
    embed_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    num_points: int = 8,
    scout_patch: int = 3,
    final_patch: int = 9,
    final_center: int = 3,
    offset_scale: float = 0.25,
    ffn_hidden: int = 256,
    enc_hidden: int = 64,
    learned_offsets: bool = True,
    scout_refine: bool = True,
    adaptive_patch: bool = True,
    share_encoder: bool = False,
    **kwargs,
  ) -> None:
    if kwargs.get("obs_normalization"):
      raise ValueError("DeltaModel: obs_normalization would fold the map in")
    if embed_dim % num_heads:
      raise ValueError(f"embed_dim {embed_dim} not divisible by {num_heads} heads")
    self.map_group = map_group
    self.L, self.W, self.coord_dim = map_shape
    self.embed_dim = embed_dim
    map_size = self.L * self.W * self.coord_dim
    active = obs_groups[obs_set]
    if map_group not in active:
      raise ValueError(f"obs_groups[{obs_set!r}] = {active} lacks {map_group!r}")
    got = int(obs[map_group].shape[-1])
    if got != map_size:
      raise ValueError(
        f"map group {map_group!r} is {got}-D but map_shape {map_shape} "
        f"implies {map_size}"
      )
    self._proprio_dim = sum(int(obs[g].shape[-1]) for g in active if g != map_group)

    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    # The query projection is per-role (actor and critic see different
    # proprio widths), so it is never shared.
    self.query_mlp = nn.Linear(self._proprio_dim, embed_dim)
    self.query_ln = nn.LayerNorm(embed_dim)

    key = _share_key(
      map_shape=map_shape,
      embed_dim=embed_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      num_points=num_points,
      scout_patch=scout_patch,
      final_patch=final_patch,
      final_center=final_center,
      offset_scale=offset_scale,
      ffn_hidden=ffn_hidden,
      enc_hidden=enc_hidden,
      learned_offsets=learned_offsets,
      scout_refine=scout_refine,
      adaptive_patch=adaptive_patch,
    )
    self.layers: nn.ModuleList
    self.base_refs: torch.Tensor
    self.cell: torch.Tensor
    self.share_encoder = share_encoder
    borrowed = _SHARED.get(key) if share_encoder and obs_set != "actor" else None
    # Only the *weights* are borrowed.  Buffers must stay owned: Module._apply
    # rebinds buffers to new tensors on .to(device) (parameters are updated in
    # place), so a borrowed buffer would still point at the owner's pre-move
    # CPU tensor and blow up on the first forward.  base_refs/cell are
    # constants, so owning a copy costs nothing and touches no gradient.
    self._borrowed_encoder = borrowed is not None
    if borrowed is not None:
      # object.__setattr__ keeps the shared weights OUT of this module's
      # _modules, so they appear once in chain(actor.params, critic.params).
      object.__setattr__(self, "layers", borrowed["layers"])
    else:
      self.layers = nn.ModuleList(
        _DeformableTerrainLayer(
          D=embed_dim,
          num_heads=num_heads,
          num_points=num_points,
          scout_patch=scout_patch,
          final_patch=final_patch,
          final_center=final_center,
          offset_scale=offset_scale,
          ffn_hidden=ffn_hidden,
          enc_hidden=enc_hidden,
          is_final=(i == num_layers - 1),
          learned_offsets=learned_offsets,
          scout_refine=scout_refine,
          adaptive_patch=adaptive_patch,
        )
        for i in range(num_layers)
      )

    # Base references: heads x points grid over the map with fixed
    # perturbations; each head owns one lateral band spanning the full length.
    xs = torch.linspace(-0.85, 0.85, num_points)
    ys = torch.linspace(-0.7, 0.7, num_heads)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H, K]
    base = torch.stack([gx, gy], dim=-1)  # [H, K, 2]
    gen = torch.Generator().manual_seed(0)
    base = base + (torch.rand(base.shape, generator=gen) * 2 - 1) * 0.05
    self.register_buffer("base_refs", base)

    # Normalized size of one map cell (align_corners=True: +-1 = edge centers).
    self.register_buffer("cell", torch.tensor([2.0 / (self.L - 1), 2.0 / (self.W - 1)]))

    if share_encoder and obs_set == "actor":
      _SHARED[key] = {
        "layers": self.layers,
        "base_refs": self.base_refs,
        "cell": self.cell,
      }

  # -- latent construction ----------------------------------------------------

  def _get_latent_dim(self) -> int:
    return self.embed_dim + self._proprio_dim

  def _encode(self, proprio: torch.Tensor, map_flat: torch.Tensor) -> torch.Tensor:
    b = proprio.shape[0]
    # Same flat layout as AME: rows = W, cols = L; only the z channel is an
    # image — x/y are the static grid, reconstructed from sample locations.
    zimg = (
      map_flat.reshape(b, self.W, self.L, self.coord_dim)[..., 2]
      .unsqueeze(1)
      .contiguous()
    )
    q = self.query_ln(self.query_mlp(proprio))
    refs = self.base_refs.unsqueeze(0).expand(b, -1, -1, -1)
    for layer in self.layers:
      q, refs = cast(_DeformableTerrainLayer, layer)(q, refs, zimg, self.cell)
    return torch.cat([q, proprio], dim=-1)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    proprio = torch.cat(
      [obs[g] for g in self.obs_groups if g != self.map_group], dim=-1
    )
    return self._encode(proprio, obs[self.map_group])

  # -- introspection / export -------------------------------------------------

  @torch.inference_mode()
  def sample_locations(self, obs) -> torch.Tensor:
    """Final-layer sample locations [B, H, K, 2] (normalized map coords)."""
    proprio = torch.cat(
      [obs[g] for g in self.obs_groups if g != self.map_group], dim=-1
    )
    b = proprio.shape[0]
    zimg = (
      obs[self.map_group]
      .reshape(b, self.W, self.L, self.coord_dim)[..., 2]
      .unsqueeze(1)
      .contiguous()
    )
    q = self.query_ln(self.query_mlp(proprio))
    refs = self.base_refs.unsqueeze(0).expand(b, -1, -1, -1)
    for layer in self.layers:
      q, refs = cast(_DeformableTerrainLayer, layer)(q, refs, zimg, self.cell)
    last = cast(_DeformableTerrainLayer, self.layers[-1])
    patch = last.final_patch if last.adaptive_patch else last.scout_patch
    return last._bound(refs, last._bounds(patch, self.cell))

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxAmeModel(self, verbose)

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("DeltaModel: use as_onnx for export")


__all__ = ["DeltaModel"]
