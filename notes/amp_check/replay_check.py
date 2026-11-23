"""Replay AMP clips (csv_to_npz output) kinematically in our G1 (mjlab g1.xml).

Checks: (1) npz joint_pos -> qpos (robot joint order) reproduces the npz body
positions (body / joint order consistent with our model); (2) npz joint_pos at
frame 0 equals the CSV row 0 mapped by joint name (csv_to_npz.py's order);
(3) foot (ankle_roll_link sole, lowest of the 15 sole points) height range;
(4) renders two frames per clip.  usage: python replay_check.py OUTDIR
"""
import os, sys
from pathlib import Path
import mujoco, mjlab, numpy as np

from ame.amp import AMP_CLIPS
from ame.mdp.foothold import sole_points_link

NPZ = Path(os.path.expanduser("~/ref/amp_mjlab_npz/amp"))
CSV = Path(os.path.expanduser("~/ref/AMP_mjlab/motion_data_csv/amp"))
CSV_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
CSV_JOINTS += ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
CSV_JOINTS += [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
m = mujoco.MjModel.from_xml_path(str(Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"))
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)][1:]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
feet = [bn.index(n) + 1 for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
P = sole_points_link().numpy()
print("model joints == csv joints (as sets):", sorted(jn) == sorted(CSV_JOINTS), "| model joint order == csv order:", jn == CSV_JOINTS)
rows = []
for c in sorted(AMP_CLIPS):
  z = np.load(NPZ / f"{c}.npz")
  T = len(z["joint_pos"])
  err = 0.0
  soles = []
  for t in range(T):
    d.qpos[:3] = z["body_pos_w"][t, 0]; d.qpos[3:7] = z["body_quat_w"][t, 0]; d.qpos[7:] = z["joint_pos"][t]
    mujoco.mj_kinematics(m, d)
    err = max(err, np.abs(d.xpos[1:] - z["body_pos_w"][t]).max())
    for f in feet:
      soles.append((d.xpos[f] + P @ d.xmat[f].reshape(3, 3).T)[:, 2].min())
  soles = np.array(soles).reshape(T, 2)
  csv0 = np.loadtxt(CSV / f"{c}.csv", delimiter=",", max_rows=1)
  by_name = dict(zip(CSV_JOINTS, csv0[7:]))
  jerr = max(abs(z["joint_pos"][0][i] - by_name[n]) for i, n in enumerate(jn))
  # lower limit check: any joint outside the model's range
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  outside = ((z["joint_pos"] < lo - 1e-3) | (z["joint_pos"] > hi + 1e-3)).mean()
  rows.append((c, T, err, jerr, soles.min(), np.percentile(soles, 1), np.median(soles.min(1)), soles.max(), outside))
print("| clip | frames | FK vs npz body pos max err (m) | frame-0 joints vs CSV by name (rad) | sole min | sole 1% | median min(L,R) | sole max | joint values outside range |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
  print(f"| {r[0]} | {r[1]} | {r[2]:.1e} | {r[3]:.1e} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:+.3f} | {r[7]:.3f} | {r[8]:.3f} |")
# render
try:
  rnd = mujoco.Renderer(m, 480, 640)
  import imageio.v3 as iio
  for c, ts in (("walk_forward_loop_002__A022", (40, 70)), ("jog_forward_loop_003__A021", (40, 52))):
    z = np.load(NPZ / f"{c}.npz")
    imgs = []
    for t in ts:
      d.qpos[:3] = z["body_pos_w"][t, 0]; d.qpos[3:7] = z["body_quat_w"][t, 0]; d.qpos[7:] = z["joint_pos"][t]
      mujoco.mj_forward(m, d)
      cam = mujoco.MjvCamera(); cam.lookat[:] = d.qpos[:3] - [0, 0, 0.35]; cam.distance = 2.6; cam.azimuth = 0; cam.elevation = -5
      rnd.update_scene(d, cam); imgs.append(rnd.render())
    iio.imwrite(out / f"{c}_frames.png", np.concatenate(imgs, axis=1))
  print("rendered")
except Exception as e:
  print("render failed:", e)
