"""Hash every registered task's env / play / rl config (functions by qualified
name, so hashes are stable across processes).  usage: python config_hashes.py OUT"""
import dataclasses, enum, hashlib, json, sys, types

import ame
from mjlab.tasks.registry import _REGISTRY as REG


def ser(o):
  if dataclasses.is_dataclass(o) and not isinstance(o, type):
    return {"__dc__": type(o).__qualname__, **{f.name: ser(getattr(o, f.name)) for f in dataclasses.fields(o)}}
  if isinstance(o, dict):
    return {str(k): ser(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [ser(v) for v in o]
  if isinstance(o, (types.FunctionType, types.BuiltinFunctionType, type, types.MethodType)):
    return f"{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', repr(o))}"
  if isinstance(o, enum.Enum):
    return str(o)
  if isinstance(o, (int, float, str, bool)) or o is None:
    return o
  if callable(o):
    return f"callable:{type(o).__module__}.{type(o).__qualname__}:{ser(vars(o)) if hasattr(o, '__dict__') else ''}"
  return f"{type(o).__qualname__}:{o!r}"


def h(o):
  return hashlib.sha256(json.dumps(ser(o), sort_keys=True, default=str).encode()).hexdigest()[:16]


if __name__ == "__main__":
 print("ame from", ame.__file__, file=sys.stderr)
 lines = []
 for name in sorted(REG):
  t = REG[name]
  lines.append(f"{name} {h(t.env_cfg)} {h(t.play_env_cfg)} {h(t.rl_cfg)}")
 open(sys.argv[1], "w").write("\n".join(lines) + "\n")
 print(len(lines), "tasks")
