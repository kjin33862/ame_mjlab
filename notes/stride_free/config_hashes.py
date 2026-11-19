"""sha256 of every registered task's env (train, play) and rl config repr.

usage: python notes/stride_free/config_hashes.py > hashes.txt
"""

import hashlib
import re
from dataclasses import asdict

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import ame  # noqa: F401


def h(x):
  return hashlib.sha256(re.sub(r" at 0x[0-9a-f]+", "", repr(x)).encode()).hexdigest()[
    :16
  ]


for t in sorted(list_tasks()):
  print(t, h(load_env_cfg(t)), h(load_env_cfg(t, play=True)), h(asdict(load_rl_cfg(t))))
