"""Unified diff of two tasks' env (train, play) and rl configs, as normalised reprs split one field per line.

usage: python notes/foothold_d30_alphanist/config_diff.py <task_a> <task_b>
"""
import difflib
import re
import sys
from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ame  # noqa: F401


def lines(x):
  s = re.sub(r" at 0x[0-9a-f]+", "", repr(x))
  return re.sub(r", (?=[A-Za-z_'\"]+[=:])", ",\n", s).splitlines()


a, b = sys.argv[1:3]
n = 0
for name, fa, fb in (
  ("env (train)", lambda: load_env_cfg(a), lambda: load_env_cfg(b)),
  ("env (play)", lambda: load_env_cfg(a, play=True), lambda: load_env_cfg(b, play=True)),
  ("rl", lambda: asdict(load_rl_cfg(a)), lambda: asdict(load_rl_cfg(b))),
):
  d = list(difflib.unified_diff(lines(fa()), lines(fb()), a, b, n=0, lineterm=""))
  print(f"=== {name}: {sum(l[0] in '+-' and l[:3] not in ('+++', '---') for l in d)} changed lines")
  print("\n".join(d[2:]))
