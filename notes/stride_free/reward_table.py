"""Print a task's reward terms (name, weight, func, params) as a markdown table.

usage: python notes/stride_free/reward_table.py <task>
"""

import sys

from mjlab.tasks.registry import load_env_cfg

import ame  # noqa: F401


def fmt(v):
  if isinstance(v, float):
    return f"{v:.6g}"
  if isinstance(v, dict):
    return "{" + ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) + "}"
  if isinstance(v, (list, tuple)):
    return "(" + ", ".join(fmt(x) for x in v) + ")"
  if hasattr(v, "name") and hasattr(v, "joint_names"):
    parts = [repr(v.name)]
    for f in ("joint_names", "body_names", "site_names", "geom_names"):
      x = getattr(v, f, None)
      if x not in (None, (), []):
        parts.append(f"{f}={fmt(x)}")
    return "SceneEntityCfg(" + ", ".join(parts) + ")"
  return str(v)


cfg = load_env_cfg(sys.argv[1])
print("| 항 | 가중치 | 함수 | 파라미터 |")
print("|---|---:|---|---|")
for name, t in cfg.rewards.items():
  func = (
    getattr(t.func, "__module__", "") + "." + getattr(t.func, "__name__", str(t.func))
  )
  params = "; ".join(f"{k}={fmt(v)}" for k, v in t.params.items()).replace("|", "\\|")
  print(f"| `{name}` | {t.weight:g} | `{func}` | {params} |")
