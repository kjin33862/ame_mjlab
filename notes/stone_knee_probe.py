"""H3 check: when a knee first touches terrain on the stone tiles, is a foot in the pit?

Runs scripts/evaluate.py's setup (ame_mjlab eval package, CPU) with the fall-only
rule, so episodes continue after a non-foot contact and the first touch is
recorded by ``illegal_contact_logged``.  Every step it keeps the pre-step state
(feet, knees); at an env's first non-foot hit it stores that state (20 ms
before the step that produced the hit) and, until the end, whether it fell.

Run from ~/ame_mjlab with its venv (the eval package lives there):
  cd ~/ame_mjlab && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    ~/ame_mjlab_sgs/notes/stone_knee_probe.py --checkpoint <pt> --terrain stone_lattice --difficulty 0.25
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "stone_knee"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--name", required=True)
  p.add_argument("--terrain", default="stone_lattice")
  p.add_argument("--difficulty", type=float, default=0.25)
  p.add_argument("--rows", type=int, default=8)
  p.add_argument("--cols", type=int, default=8)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--seconds", type=float, default=8.0)
  p.add_argument("--device", default="cpu")
  args = p.parse_args()

  from ame.eval import (
    CONTACT_BODY_ATTR,
    CONTACT_HIT_ATTR,
    OFF_HEIGHT_NOISE,
    assign_tiles,
    checkpoint_actor_obs,
    eval_env_cfg,
  )
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  device = args.device
  task = "Ame-Velocity-Rough-G1"
  cfg = eval_env_cfg(
    terrain=args.terrain,
    num_rows=args.rows,
    num_cols=args.cols,
    difficulty=args.difficulty,
    seed=args.seed,
    height_noise=dict(OFF_HEIGHT_NOISE),
    proprio_noise=False,
    actor_obs=checkpoint_actor_obs(args.checkpoint),
    fail_rule="fall",
  )
  rl_cfg = load_rl_cfg(task)
  rl_cfg.seed = args.seed
  t0 = time.time()
  base = ManagerBasedRlEnv(cfg=cfg, device=device)
  assign_tiles(base)
  env = RslRlVecEnvWrapper(base, clip_actions=rl_cfg.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    env, asdict(rl_cfg), device=device
  )
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  print(f"build {time.time() - t0:.0f}s")

  robot = base.scene["robot"]
  names = robot.body_names
  idx = {
    n: names.index(n)
    for n in (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_knee_link",
      "right_knee_link",
      "pelvis",
    )
  }
  sensor = base.scene[cfg.terminations["illegal_contact"].params["sensor_name"]]
  prim = list(sensor.primary_names)
  top = 0.0 if args.terrain == "stone_lattice" else 0.2
  pit_below = 0.05  # a foot whose ankle-link origin is this far below the stone-top plane + 0.035 sole offset

  n = base.num_envs
  first_step = torch.full((n,), -1, dtype=torch.long)
  rec: dict[str, torch.Tensor] = {}
  fell = torch.zeros(n, dtype=torch.bool)
  ended = torch.zeros(n, dtype=torch.bool)  # first episode over
  foot_in_pit_ever = torch.zeros(n, 2, dtype=torch.bool)
  first_pit_step = torch.full((n, 2), -1, dtype=torch.long)

  obs, _ = env.reset()
  steps = int(args.seconds / base.step_dt)
  for t in range(steps):
    pos = robot.data.body_link_pos_w.cpu()
    feet_z = torch.stack(
      [pos[:, idx["left_ankle_roll_link"], 2], pos[:, idx["right_ankle_roll_link"], 2]],
      -1,
    )
    knee_z = torch.stack(
      [pos[:, idx["left_knee_link"], 2], pos[:, idx["right_knee_link"], 2]], -1
    )
    sole_z = feet_z - 0.035  # flat-foot sole
    in_pit = sole_z < top - pit_below
    newly = in_pit & ~foot_in_pit_ever & ~ended[:, None]
    first_pit_step[newly] = t
    foot_in_pit_ever |= in_pit & ~ended[:, None]
    state = {
      "sole_z": sole_z - top,
      "knee_z": knee_z - top,
      "pelvis_z": pos[:, idx["pelvis"], 2] - top,
    }
    with torch.no_grad():
      act = policy(obs)
    obs, _, dones, _ = env.step(act)
    hit = getattr(base, CONTACT_HIT_ATTR, torch.zeros(n, dtype=torch.bool)).cpu()
    first = hit & (first_step < 0) & ~ended
    for k, v in state.items():
      rec.setdefault(k, torch.full((n, *v.shape[1:]), float("nan")))
      rec[k][first] = v[first].cpu()
    first_step[first] = t
    d = dones.cpu().bool()
    newly_done = d & ~ended
    if newly_done.any():
      to = base.reset_time_outs.cpu().bool()
      fell |= newly_done & ~to
    ended |= d
    if bool(ended.all()):
      break

  body = getattr(base, CONTACT_BODY_ATTR, torch.full((n,), -1)).cpu()
  out = []
  for i in range(n):
    if first_step[i] < 0:
      out.append(
        {
          "env": i,
          "hit": False,
          "fell": bool(fell[i]),
          "foot_in_pit_ever": foot_in_pit_ever[i].tolist(),
        }
      )
      continue
    b = prim[int(body[i])] if int(body[i]) >= 0 else "?"
    side = 0 if b.startswith("left") else (1 if b.startswith("right") else -1)
    fp = first_pit_step[i].tolist()
    out.append(
      {
        "env": i,
        "hit": True,
        "body": b,
        "t_s": round(float(first_step[i]) * base.step_dt, 2),
        "sole_z_lr": [round(float(x), 3) for x in rec["sole_z"][i]],
        "knee_z_lr": [round(float(x), 3) for x in rec["knee_z"][i]],
        "pelvis_z": round(float(rec["pelvis_z"][i]), 3),
        "same_side_foot_in_pit": bool(
          side >= 0 and rec["sole_z"][i][side] < -pit_below
        ),
        "any_foot_in_pit": bool((rec["sole_z"][i] < -pit_below).any()),
        "foot_pit_before_hit": [bool(0 <= s <= int(first_step[i])) for s in fp],
        "fell": bool(fell[i]),
      }
    )
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"{args.name}_{args.terrain}_d{args.difficulty}_s{args.seed}.json"
  path.write_text(json.dumps(out, indent=0))
  hits = [o for o in out if o["hit"]]
  knee = [o for o in hits if "knee" in o["body"]]
  print(
    f"{args.name} {args.terrain} d{args.difficulty}: {len(hits)}/{n} hit, knee {len(knee)}, "
    f"same-side foot in pit at knee hit {sum(o['same_side_foot_in_pit'] for o in knee)}, "
    f"any foot in pit {sum(o['any_foot_in_pit'] for o in knee)}, fell {sum(o['fell'] for o in out)}, "
    f"no hit but foot in pit {sum(any(o['foot_in_pit_ever']) for o in out if not o['hit'])}"
  )
  print("wrote", path)


if __name__ == "__main__":
  main()
