"""Probe: sole-scan hits / frame pose read inside the reward call vs after env.step (4 envs, 8 steps, random actions).

usage: python notes/foothold_v2_check/hit_timing_probe.py
"""
import torch, ame
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from ame.mdp.foothold_v2 import foothold_v2
T="Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1"
c=load_env_cfg(T); c.scene.num_envs=4; c.events.pop("push_robot"); c.curriculum={}
env=ManagerBasedRlEnv(c,device="cuda:0"); env.reset()
rm=env.reward_manager; term=rm._term_cfgs[rm._term_names.index("foothold_v2")].func
sole=env.scene["foot_sole_scan"]; robot=env.scene["robot"]
at_reward=[]
orig=term.__call__
class Wrap:
  def reset(self, env_ids=None): term.reset(env_ids)
  def __call__(self,*a,**k):
    at_reward.append((sole.data.hit_pos_w[0,:,:2].clone(), sole.data.frame_pos_w[0].clone()))
    return orig(*a,**k)
rm._term_cfgs[rm._term_names.index("foothold_v2")].func=Wrap()
after=[]
a=torch.randn(4,env.action_manager.total_action_dim,device="cuda:0")*0.3
for s in range(8):
  env.step(a)
  after.append((sole.data.hit_pos_w[0,:,:2].clone(), sole.data.frame_pos_w[0].clone(), robot.data.body_link_pos_w[0,6,:2].clone()))
for s in range(1,8):
  r=at_reward[s][1]; print(s, "reward-time frame vs after same step: %.4f"%(r-after[s][1]).abs().max().item(), " vs after previous step: %.4f"%(r-after[s-1][1]).abs().max().item())
for s in range(1,8):
  r=at_reward[s][0]; print(s, "HITS reward-time vs after same step: %.4f"%(r-after[s][0]).abs().max().item(), " vs after previous step: %.4f"%(r-after[s-1][0]).abs().max().item())
