"""AME's terrain attention and the actor/critic built on it.

Port of ``ActorCriticEncoder`` from SII-FUSC/AME_Locomotion onto rsl-rl's
model API.  The encoder is the reference's: the elevation scan (per-point
yaw-frame coordinates) through a small CNN into a grid of ``mha_dim`` tokens,
attended by a single query embedded from proprioception, the attended feature
concatenated with the raw proprio for the MLP.  What changed is framing only:
the map is its own observation group instead of a slice off the tail of one
flat vector, and actor and critic are two instances of one model class.

:class:`TerrainAttention` is the shared piece; :class:`AmeModel` is the
from-scratch actor/critic; :class:`TerrainContext` hands the same encoder to
another actor network as a context branch.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from ame.scan import MAP_GROUP, MAP_SHAPE

if TYPE_CHECKING:
  from ame.delta import DeltaModel

# Actor/critic encoder sharing (AME tuning guide: the critic's better
# supervision speeds convergence).  The actor owns the module; the critic binds
# the SAME object via object.__setattr__ so it stays out of its _modules —
# rsl-rl builds one optimizer over chain(actor.parameters(), critic.parameters())
# and a doubly-registered parameter would be stepped twice.
_SHARED: dict[str, nn.Module] = {}


def _shared_key(map_shape, mha_dim, num_heads, cnn_downsample) -> str:
  return repr((tuple(map_shape), mha_dim, num_heads, cnn_downsample))


class TerrainAttention(nn.Module):
  """CNN tokens over the scan, attended by one ``mha_dim`` query."""

  def __init__(
    self,
    map_shape: tuple[int, int, int] = MAP_SHAPE,
    mha_dim: int = 64,
    num_heads: int = 16,
    cnn_downsample: bool = True,
  ) -> None:
    super().__init__()
    self.L, self.W, self.coord_dim = map_shape
    self.mha_dim = mha_dim
    if cnn_downsample:
      self.map_cnn = nn.Sequential(
        nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, mha_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(mha_dim),
      )
    else:
      self.map_cnn = nn.Sequential(
        nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, mha_dim, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(mha_dim),
      )
    self.mha = nn.MultiheadAttention(
      embed_dim=mha_dim, num_heads=num_heads, batch_first=True
    )

  def tokens(self, map_flat: torch.Tensor) -> torch.Tensor:
    """``[B, L*W*3]`` → ``[B, T, mha_dim]``.  The grid sensor flattens
    meshgrid(indexing="xy"): rows = W, cols = L, the reference's CNN layout."""
    height_map = map_flat.reshape(-1, self.W, self.L, self.coord_dim)
    tokens = self.map_cnn(height_map.permute(0, 3, 1, 2))
    return tokens.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1, self.mha_dim)

  def attend(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """``query [B, mha_dim]`` over ``tokens`` → ``[B, mha_dim]``."""
    attended, _ = self.mha(query=query.unsqueeze(1), key=tokens, value=tokens)
    return attended.squeeze(1)

  def forward(self, query: torch.Tensor, map_flat: torch.Tensor) -> torch.Tensor:
    return self.attend(query, self.tokens(map_flat))


def shared_terrain_attention(obs_set: str, share: bool, **kwargs) -> TerrainAttention:
  """The actor builds and registers the encoder; a sharing critic borrows it."""
  key = _shared_key(
    kwargs.get("map_shape", MAP_SHAPE),
    kwargs.get("mha_dim", 64),
    kwargs.get("num_heads", 16),
    kwargs.get("cnn_downsample", True),
  )
  if share and obs_set != "actor" and key in _SHARED:
    return cast(TerrainAttention, _SHARED[key])
  module = TerrainAttention(**kwargs)
  if share and obs_set == "actor":
    _SHARED[key] = module
  return module


def _bind(model: nn.Module, name: str, module: nn.Module, owned: bool) -> None:
  """Register ``module`` as a child when owned, else alias it unregistered."""
  if owned:
    setattr(model, name, module)
  else:
    object.__setattr__(model, name, module)


class AmeModel(MLPModel):
  """MLP actor/critic with the AME terrain attention in front.

  The base class sizes the MLP from the latent (attended map feature ⊕
  proprio), so the geometry attributes exist before ``super().__init__``.
  """

  terrain: TerrainAttention
  global_encoder: nn.Module
  query_projector: nn.Module

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    map_group: str = MAP_GROUP,
    map_shape: tuple[int, int, int] = MAP_SHAPE,
    mha_dim: int = 64,
    num_heads: int = 16,
    cnn_downsample: bool = True,
    attach_global: bool = False,
    share_encoder: bool = False,
    **kwargs,
  ) -> None:
    if kwargs.get("obs_normalization"):
      raise ValueError(
        "AmeModel does not support obs_normalization: the base normalizer "
        "would fold the elevation map into its statistics; the reference "
        "trained with normalization off."
      )
    self.map_group = map_group
    self.L, self.W, self.coord_dim = map_shape
    self.mha_dim = mha_dim
    self.attach_global = attach_global
    active = obs_groups[obs_set]
    if map_group not in active:
      raise ValueError(f"obs_groups[{obs_set!r}] = {active} lacks {map_group!r}")
    map_size = self.L * self.W * self.coord_dim
    got = int(obs[map_group].shape[-1])
    if got != map_size:
      raise ValueError(
        f"map group {map_group!r} is {got}-D but map_shape {map_shape} "
        f"implies {map_size}; the sensor grid and the model disagree"
      )
    self._proprio_dim = sum(int(obs[g].shape[-1]) for g in active if g != map_group)

    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    # Per role, never shared: the actor and critic proprio widths differ.
    self.proprio_embedding = nn.Linear(self._proprio_dim, mha_dim)
    owned = not share_encoder or obs_set == "actor"
    terrain = shared_terrain_attention(
      obs_set,
      share_encoder,
      map_shape=map_shape,
      mha_dim=mha_dim,
      num_heads=num_heads,
      cnn_downsample=cnn_downsample,
    )
    _bind(self, "terrain", terrain, owned)
    self._register_load_state_dict_pre_hook(self._upgrade_legacy_keys)
    if attach_global:
      # ame2: a global max-pool context folded into the query.
      from rsl_rl.modules import MLP

      key = _shared_key(map_shape, mha_dim, num_heads, cnn_downsample) + "/global"
      if owned or key not in _SHARED:
        both = nn.ModuleDict(
          {
            "global_encoder": MLP(mha_dim, mha_dim, [256, 128], "elu"),
            "query_projector": nn.Linear(mha_dim * 2, mha_dim),
          }
        )
        if share_encoder and obs_set == "actor":
          _SHARED[key] = both
      else:
        both = cast(nn.ModuleDict, _SHARED[key])
      _bind(self, "global_encoder", both["global_encoder"], owned)
      _bind(self, "query_projector", both["query_projector"], owned)

  def _get_latent_dim(self) -> int:
    return self.mha_dim * (2 if self.attach_global else 1) + self._proprio_dim

  @staticmethod
  def _upgrade_legacy_keys(state_dict: dict, prefix: str, *args) -> None:
    """Checkpoints before the ``TerrainAttention`` split kept ``map_cnn`` and
    ``mha`` directly on the model."""
    del args
    for key in list(state_dict):
      tail = key[len(prefix) :]
      if tail.startswith(("map_cnn.", "mha.")):
        state_dict[f"{prefix}terrain.{tail}"] = state_dict.pop(key)

  def _encode(self, proprio: torch.Tensor, map_flat: torch.Tensor) -> torch.Tensor:
    """Attended terrain feature ⊕ proprio (⊕ global max-pool feature)."""
    tokens = self.terrain.tokens(map_flat)
    query = self.proprio_embedding(proprio)
    if not self.attach_global:
      return torch.cat([self.terrain.attend(query, tokens), proprio], dim=-1)
    global_max, _ = torch.max(self.global_encoder(tokens), dim=1)
    query = self.query_projector(torch.cat([global_max, query], dim=-1))
    return torch.cat([global_max, self.terrain.attend(query, tokens), proprio], dim=-1)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    proprio = torch.cat(
      [obs[g] for g in self.obs_groups if g != self.map_group], dim=-1
    )
    return self._encode(proprio, obs[self.map_group])

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxAmeModel(self, verbose)

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("AmeModel: use as_onnx for export")


class TerrainContext(nn.Module):
  """AME's terrain attention as a context branch for a foreign actor.

  The query is embedded from ``query_groups`` (every non-map group the actor
  sees by default);
  ``forward(obs)`` returns the attended ``[B, mha_dim]`` feature.  With
  ``share_encoder`` the encoder is registered for an ``AmeModel`` critic to
  borrow, exactly as the from-scratch actor does.
  """

  terrain: TerrainAttention

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    map_group: str = MAP_GROUP,
    map_shape: tuple[int, int, int] = MAP_SHAPE,
    mha_dim: int = 64,
    num_heads: int = 16,
    cnn_downsample: bool = True,
    share_encoder: bool = True,
    query_groups: tuple[str, ...] | None = None,
  ) -> None:
    super().__init__()
    self.map_group = map_group
    self.query_groups = (
      list(query_groups)
      if query_groups is not None
      else [g for g in obs_groups[obs_set] if g != map_group]
    )
    query_dim = sum(int(obs[g].shape[-1]) for g in self.query_groups)
    self.query = nn.Linear(query_dim, mha_dim)
    self.out_dim = mha_dim
    terrain = shared_terrain_attention(
      obs_set,
      share_encoder,
      map_shape=map_shape,
      mha_dim=mha_dim,
      num_heads=num_heads,
      cnn_downsample=cnn_downsample,
    )
    self.shared = share_encoder
    _bind(self, "terrain", terrain, owned=not share_encoder or obs_set == "actor")

  def adapted_parameters(self) -> Iterator[nn.Parameter]:
    """The query embedding always; the encoder only when it is this branch's
    alone (a shared one trains under the critic's loss from the start)."""
    yield from self.query.parameters()
    if not self.shared:
      yield from self.terrain.parameters()

  def forward(self, obs: TensorDict) -> torch.Tensor:
    query = self.query(torch.cat([obs[g] for g in self.query_groups], dim=-1))
    return self.terrain(query, obs[self.map_group])


class _OnnxAmeModel(nn.Module):
  """Deterministic ``(proprio, height_map) -> action`` wrapper for export.
  Also serves ``DeltaModel``, which keeps the same geometry attributes."""

  def __init__(self, model: AmeModel | DeltaModel, verbose: bool = False) -> None:
    super().__init__()
    del verbose
    self.model = model
    self.input_names = ["proprio", "height_map"]
    self.output_names = ["action"]

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    m = self.model
    return torch.zeros(1, m._proprio_dim), torch.zeros(1, m.L * m.W * m.coord_dim)

  def forward(self, proprio: torch.Tensor, height_map: torch.Tensor) -> torch.Tensor:
    out = self.model.mlp(self.model._encode(proprio, height_map))
    if self.model.distribution is not None:
      out = self.model.distribution.deterministic_output(out)
    return out


__all__ = ["AmeModel", "TerrainAttention", "TerrainContext", "shared_terrain_attention"]
