코드 커밋 `1639ce5` (branch `goal-reach`) — 학습 중: **Part B `Ame-Mjlab-Gap-Edge-AlphaNIST-Contact-FootholdD30-GoalReach-Rough-G1`** (run `goal_reach_1024`, 1024 envs, seed 42, 10000 it)

# GoalReach 보고서 (2026-11-21 새벽)

## 0. 결론 먼저

- **Part B(AME-2식 goal reaching)가 테스트·smoke·런타임 검사를 모두 통과해서 Part B를 학습 중**입니다. Part A(AR025-Stag)도 등록·테스트·smoke까지 끝나 있어서 바로 대체 실행 가능합니다(§7).
- 기존 30개 태스크의 env(train/play)·agent config 해시는 51b5d6f와 **완전히 동일**합니다(§5).
- 공식 코드와 다른 부분은 대부분 **보드 차이(8 m 타일, 5개 family, flat-patch 샘플러 없음)** 와 **우리 쪽에서 유지하기로 한 것(SGS, foothold+double critic, mjlab 정규화 항, ame_contact 종료)** 때문입니다. 보상 4개·stagnation·종료 패널티·관측 포맷·성공 기준·yaw/cone 커리큘럼 스케줄은 공식 수식·값 그대로입니다(§2).
- 속도: 2080 Ti에서 **약 10 s/iteration** → 10k it는 약 28 시간. 아침(약 8 시간 후)에는 **약 2.5–3k iteration**이 되어 있을 것입니다. 1k–2k 구간의 정상/실패 판정 기준은 §8.
- 주의할 점 하나: 공식 stagnation은 **3-D** root 변위를 씁니다. 그래서 넘어지는 동작(루트 0.75 m → 0.1 m) 자체가 첫 5 s 창에서는 "움직임"으로 셉니다. 우리 태스크에서는 넘어지면 contact 종료가 먼저 걸리므로 실제 영향은 없고, 런타임 검사에서 이 동작을 확인했습니다(§6).

## 1. Step 0 — 저장소 확인

```
$ git status                 # HEAD detached at mine/stride-free, untracked: .claude/, notes/foothold_eval/foothold_m9999/
$ git log -1 --oneline       # 51b5d6f Add StrideFree report: reward table, changes, config diff, checks and the 2080 Ti resume command
$ git remote -v
mine    https://github.com/kjin33862/ame_mjlab.git (fetch)
mine    https://github.com/kjin33862/ame_mjlab.git (push)
origin  https://github.com/psrobotics/ame_mjlab.git (fetch)
origin  no_push (push)
```
HEAD가 51b5d6f였으므로 fetch 없이 `git checkout -b goal-reach 51b5d6f`. 이 머신에는 git 사용자 정보가 없어서, 커밋은 기존 커밋과 같은 작성자를 `git -c user.name=… -c user.email=…`로 지정해 만들었습니다(전역 설정은 바꾸지 않았습니다).

## 2. 공식 코드(ame2_minimal) 표

출처: `github.com/leggedrobotics/ame2_minimal` @ `34b6fe9` (2026-10-30), 읽기 전용으로 clone. 경로는 `ame2/ame2/tasks/` 기준. dt = sim 0.005 × decimation 4 = **0.02 s** (`ame2_env_cfg.py:674-677`). Isaac Lab과 mjlab 모두 reward에 dt를 곱하므로 weight는 그대로 옮길 수 있습니다.

| 항목 | 공식 값 / 수식 | 파일:줄 | 우리 구현 |
|---|---|---|---|
| 에피소드 길이 | 16.0 s (800 step) | `ame2_env_cfg.py:675` | 16.0 s (기존 20 s) |
| 목표 개수 | 에피소드당 1개: `resampling_time_range=(16.01,16.01)` | `ame2_env_cfg.py:153-162` | resampling 1e9 s, reset 때만 샘플 |
| 목표 방향 | env origin 기준 forward cone `\|y\| <= x·tan(cone)+0.1` 안의 flat patch 우선, 없으면 아무 patch | `mdp/commands.py:25-55` | cone 안에서 방향을 직접 샘플 (§3-1) |
| 목표 높이 | patch z + default root z | `mdp/commands.py:58` | 타일 중심 z + default root z (보상·관측에 안 쓰임) |
| 목표 heading | `simple_heading=False` → world heading U(−π/3, π/3) | `ame2_env_cfg.py:156-161`, `mdp/commands.py:77-80` | 동일 |
| 목표 거리 (family별 flat-patch 범위, 12 m 타일, origin 기준) | rough x 0–5.5, y ±5.5 / stair_down·stair_up x 1–6, **y ±0.2** / boxes x 0–5.5 / obstacles x 2–5.5 / climb_up·climb_down x 2.6–5.5, y ±3 / consecutive x 4–5 / gap x 1.8–5 / pallets x 3.5–5.8 / beam x 4–5.8, y ±0.5 / **stones x 1–5.8, y ±5** | `terrains.py:24-246` | stones·wave·rough d∈[1.0, 3.5] m, stairs d∈[2.0, 3.5] m 직진(+x, 횡 ±0.2 m), Chebyshev ≤ 3.5 m, lattice는 가장 가까운 돌 중심으로 snap |
| 명령 | `(x_b, y_b, z_b, heading_err)` (yaw frame) | Isaac Lab `TerrainBasedPose2dCommand` (`mdp/commands.py:11,18`) | 동일 (`GoalCommand`) |
| actor 목표 관측 | `[x, y, sin e, cos e]`, `dist_thr=2.0`: 2 m 넘으면 위치는 2 m로 축소, e는 매 step U(−π, π) 랜덤 | `ame2_env_cfg.py:259-262`, `mdp/observations.py:263-298` | 동일 (`goal_obs(dist_clip=2.0)`) |
| actor 남은 시간 | **없음** ("the teacher actor does not observe remaining episode time") | `ame2_env_cfg.py:262` | 없음 |
| critic 목표 관측 | 같은 포맷, clip 없음 + `time_left = 1 − t/T` | `ame2_env_cfg.py:201-204`, `mdp/observations.py:253-255` | 동일 (command 바로 뒤에 `time_left`) |
| sinusoidal encoding | 없음 (sin/cos of heading error만) | `mdp/observations.py:291-298` | 없음 |
| position tracking | `100 · 1/(1+(d/2)^2) · 1(t_left<4)/4` | `ame2_env_cfg.py:413-415`, `mdp/rewards.py:219-236` | 동일 |
| heading tracking | `50 · 1/(1+e^2) · 1(t_left<2)/2 · 1(d<0.5)` | `ame2_env_cfg.py:416-418`, `mdp/rewards.py:239-259` | 동일 |
| move-to-goal | `5 · 1[d<0.5 or (cos>0.5 and 0.3≤\|v\|≤2.0)]`, v = torso_link 속도를 root quat로 돌린 xy | `ame2_env_cfg.py:419-428`, `mdp/rewards.py:180-216` | 동일 (torso_link) |
| stand-at-goal | `5 · 1[d<0.5 and \|e\|<0.5] · exp(−(d_foot+d_g+d_q+d)/4)`, d_foot=비접촉 발 비율, d_g=1−g_z², d_q=mean\|q−q0\| | `ame2_env_cfg.py:429-436`, `mdp/rewards.py:262-304` | 동일 (발 접촉은 mjlab `feet_ground_contact.found`) |
| 조기 종료 패널티 | weight **−500 = −10/dt**, `early_.*` 종료 항 개수 합 (time-out 제외) → 종료 1회당 −10 | `ame2_env_cfg.py:439-443`, `mdp/rewards.py:35-61` | `termination_penalty` −50 → **−500** (mjlab `is_terminated`, 0/1) |
| action smoothness | −0.01 · Σ(a−a_prev)² | `ame2_env_cfg.py:520-523`, `mdp/rewards.py:174-176` | `action_rate_l2` −0.1 → **−0.025** (지시값) |
| stagnation 종료 | 5 s마다 root pos(3-D) snapshot과 비교, 이동 < 0.5 m 이고 목표까지 > 1.0 m면 종료; snapshot은 `episode_length_buf<=1`과 매 검사 때 갱신 | `ame2_env_cfg.py:605-613`, `mdp/terminations.py:180-230` | 동일 (`goal_stagnation`, 창 = 250 step) |
| 성공 기준 | 에피소드 끝에서 `d < 0.5 and t_left < 4` | `mdp/curriculums.py:314-317` | 동일 |
| 지형 promote/demote | EMA ← 0.8·EMA + 0.2·success; up = `d<0.5 and EMA>0.5`; down = `d>4.0` | `ame2_env_cfg.py:620-623`, `mdp/curriculums.py:292-327` | 사다리 대신 SGS (AlphaNIST + 20 % uniform), success만 공식 기준 |
| 초기 지형 레벨 | `max_init_terrain_level=5` | `ame2_env_cfg.py:51` | 0–2 (기존), 이후 SGS |
| heading 초기화 커리큘럼 | spawn yaw 범위 0 → ±π, `max_steps = 24·2000` step 선형 (cfg 초기값 ±π/4는 step 0에서 0으로 덮임) | `ame2_env_cfg.py:374-388, 624-631`, `mdp/curriculums.py:175-187` | 동일 스케줄 (`yaw_schedule`) |
| goal cone 커리큘럼 | 10° → 45°, `24·2000` step 선형 | `ame2_env_cfg.py:632-639`, `mdp/curriculums.py:190-204` | 동일 |
| 대칭 증강 | goal `(x, −y, −sin, cos)`, time_left 불변 | `mdp/symmetry.py:182-189` | 동일 (`mirror_obs_actions_goal`) |
| rollout | `num_steps_per_env=24` | `agents/rsl_rl_ppo_cfg.py:149` | 24 (기존) |

## 3. 공식 코드와 다른 점과 이유

1. **목표 샘플링 방식.** 공식은 family마다 flat-patch(`target`)를 미리 뽑아 두고 cone 안의 patch를 고릅니다. 우리 gap board에는 `target` patch 샘플러가 없고 타일이 8 m(공식 12 m)이라 해석적으로 샘플합니다: 방향 θ ~ U(−cone, cone) (world +x 기준, 공식과 같은 축), 거리 d ~ U(범위), 타일 중심으로부터 Chebyshev 3.5 m(타일 가장자리 4 m에서 0.5 m 안쪽)를 넘으면 축소, stepping stones는 가장 가까운 돌 중심으로 snap. 목표가 항상 같은 타일 안의 서 있을 수 있는 곳에 놓이도록 하기 위해서입니다. 공식은 cone 밖 fallback도 있지만, 우리는 항상 cone 안입니다.
2. **거리 범위.** 공식은 stones x 1–5.8 m (12 m 타일). 우리는 반 타일이 4 m라 stones·wave·rough 1.0–3.5 m. 계단(platform 3 m, 계단 구간 중심에서 1.5–3.0 m, 그 바깥 1 m가 평지)은 2.0–3.5 m 직진(+x, 횡 ±0.2 m, 공식 stair patch의 y ±0.2와 같음)이라, 목표가 아래/위쪽 계단이나 바깥 평지에 놓이고 계단을 지나야 합니다.
3. **spawn.** 공식은 origin ±0.2 m + 초기 속도 랜덤(±0.5). 우리는 기존 centre spawn(0.5 m 원판, lattice에서는 돌 위)과 기존 관절 reset을 그대로 쓰고, yaw만 공식 커리큘럼으로 바꿨습니다(기존 uniform ±π → 0에서 ±π로 2000 it에 걸쳐 증가). 기존 태스크의 spawn은 바뀌지 않습니다(`yaw_schedule`가 없으면 이전 동작).
4. **actor 관측 세트.** 공식 teacher actor에는 base_lin_vel이 있습니다. 우리는 기존 배포용 actor 세트(lin_vel 없음)에 command 자리만 goal 4차원으로 바꿨습니다. critic은 기존 privileged 세트 + goal(clip 없음) + time_left.
5. **종료 패널티의 단위.** 공식은 동시에 걸린 `early_*` 항 개수를 더하고, mjlab `is_terminated`는 0/1입니다. 차이는 같은 step에 두 항이 동시에 걸릴 때뿐입니다. 종료 항 자체도 다릅니다: 우리는 `illegal_contact`(발 외 접촉 > 1 N, ame_contact), `fell_over`, `stagnation`이 패널티 대상이고 `out_of_terrain_bounds`·`time_out`은 time-out입니다. 공식은 gravity, hip 가속도, torso/pelvis 400 N 접촉, stagnation입니다. contact 종료는 지시대로 유지했습니다.
6. **정규화 항.** 공식 AME-2 패널티 세트 대신 기존 mjlab G1 세트(upright, pose, body_ang_vel, angular_momentum, dof_pos_limits, foot_clearance/swing_height(one-sided), foot_slip, soft_landing, self_collisions, undesired_contacts, foothold)를 유지했습니다(지시: "all other regularizers" 유지). `action_rate_l2`는 공식 −0.01이 아닌 지시값 −0.025.
7. **`gait` 파생 명령 (공식에 없음).** 기존 항 중 pose(자세 regime 선택), foot_clearance, foot_swing_height, foot_slip, soft_landing, air_time(weight 0), swing 진단 metric이 `twist` 명령의 크기로 on/off 됩니다. twist를 없애면 이들이 깨지므로, 관측에는 들어가지 않는 파생 twist `gait`를 추가했습니다: 목표가 0.5 m 밖이면 목표 방향 단위벡터 × 1.0 m/s, 안이면 0 + yaw rate = clip(heading error, ±0.4). 결과적으로 이동 중에는 정규화가 켜지고 pose는 walking std, 목표에 도착해서 방향이 맞으면 standing std(stand_at_goal과 같은 방향)가 됩니다. running regime(≥1.5)은 쓰지 않습니다.
8. **커리큘럼.** 공식 사다리(EMA>0.5이고 d<0.5면 up, d>4면 down) 대신 지시대로 기존 SGS(AlphaNIST + 20 % uniform)를 유지하고 success만 공식 기준(`d<0.5 and t_left<4`)으로 바꿨습니다. EMA 게이트는 SGS 버퍼(H=100)가 그 역할을 하므로 두지 않았습니다. d는 reset 시점의 root 위치와 목표 world 위치로 다시 계산합니다(공식은 직전 step의 command 버퍼라 최대 1 step 차이).
9. **teacher mapping noise 커리큘럼**(공식 `ame2_env_cfg.py:640-647`)은 공식 매핑 파이프라인 전용이라 옮기지 않았고, 기존 scan noise를 그대로 둡니다.
10. **double critic + foothold(δ=0.30)** 는 우리 쪽 구성으로 유지했습니다(group 2 = foothold, A = 1.0·A1 + 0.25·A2).
11. **stagnation 창**은 시간 대신 step 수(round(5/0.02)=250)로 셉니다. 부동소수점 비교로 한 step이 밀리는 일을 막기 위해서이고, 의미는 같습니다.

## 4. 변경 파일

| 파일 | 목적 |
|---|---|
| `src/ame/mdp/goal.py` (신규) | GoalCommand / gait 파생 명령, goal 관측·time_left, 보상 4개, `goal_stagnation`, Part A용 `velocity_stagnation`, `goal_distance` metric, `terrain_levels_sgs_goal`(성공 = goal reached + 로그) |
| `src/ame/mdp/events.py` | centre spawn에 선택적 `yaw_schedule`(없으면 기존 uniform ±π 그대로) |
| `src/ame/mdp/__init__.py` | goal 모듈 export |
| `src/ame/env_cfg.py` | Part A: `apply_action_rate_weight`, `apply_velocity_stagnation`, `ame_gap_ar_stag_env_cfg`; Part B: `apply_goal_task`, `apply_goal_success`, `ame_goal_env_cfg` |
| `src/ame/symmetry.py` | 공통 `_mirror_group`로 정리(기존 velocity mirror 동작 동일), 그룹 이름으로 레이아웃을 정하는 `mirror_obs_actions_goal` |
| `src/ame/rl_cfg.py` | `goal_double_critic_ppo_runner_cfg` (double critic + goal mirror) |
| `src/ame/__init__.py` | 두 태스크 등록 + docstring |
| `tests/test_goal_reach.py` (신규) | 18개 테스트: 기존 해시 불변, Part A config 차이 2개뿐, stagnation 두 종류, goal config 전체, 보상 4개 수치, goal 관측 clip/랜덤, gait 명령, 대칭(부호·involution·mirrored world 일치·velocity mirror 불변), 스케줄 |
| `notes/goal_reach/config_hashes.py`, `config_hashes_51b5d6f.txt`, `config_hashes_new.txt` | 해시 비교 (기준 / 변경 후) |
| `notes/goal_reach/config_diff.py`, `config_diff_ar025_stag.txt` | Part A와 base의 config diff |
| `notes/goal_reach/goal_runtime_check.py`, `.txt` | 실제 env에서 목표 배치·관측·stagnation·패널티 검사 |
| `notes/goal_reach/pytest.txt` | 전체 테스트 결과 |
| `notes/goal_reach/report.md` | 이 보고서 |

## 5. 테스트·smoke 결과

- **전체 테스트**: `uv run pytest -q` → **138 passed, 1 skipped** (기존 120 + 신규 18; skip은 기존 것).
- **기존 config 해시**: `notes/goal_reach/config_hashes.py` 결과를 51b5d6f 기준(30개 태스크)과 비교 → 30개 모두 env(train)·env(play)·agent 해시 동일, 새 줄 2개만 추가. 기준값은 StrideFree 보고서의 `config_hashes_new.txt`와도 같습니다. 테스트 `test_existing_task_config_hashes_unchanged`가 이를 계속 검사합니다.
- **Part A config diff** (`config_diff_ar025_stag.txt`): train·play 모두 `action_rate_l2` −0.1 → −0.025와 `stagnation` 항 추가뿐, agent cfg 차이 0.
- **ruff**: 새·수정 파일 check/format 깨끗 (기존 test_edge/test_foothold/test_sgs의 format 차이는 원래 있던 것이라 건드리지 않음).
- **Smoke (64 envs × 3 it, seed 42, tensorboard)**:
  - Part A: exit 0, 3 iteration, NaN 없음, `Episode_Termination/stagnation` 태그 생성, iteration 1.7 s.
  - Part B: exit 0, 3 iteration, NaN 없음, iteration 1.8 s. 마지막 iteration: Mean reward −9.95, termination_penalty −0.625 (= 모든 에피소드가 −10으로 종료, ÷16 s), goal/contact 1.0(랜덤 정책은 다 넘어짐), final_distance 2.50 m, pelvis_speed 0.30, cone 10.0°, yaw_limit 0.004 rad.

## 6. 런타임 검사 (`goal_runtime_check.txt`, 64 envs, GPU)

- 목표 위치(reset 직후, cone 10°): stones d 1.15–3.44 m, stairs 2.02–3.48 m (방위 ±5° 이내 = 직진 + 0.2 m 횡), wave/rough 1.1–3.4 m; Chebyshev 최대 3.48 m; lattice 목표의 돌 중심까지 거리 0; heading −58°–+59° (±60° 안).
- actor goal = critic goal을 2 m로 clip한 것과 일치, 2 m 안에서는 sin/cos도 일치; time_left는 reset 때 1.0, step마다 1/800 감소.
- stagnation: 영 행동이면 로봇이 1 s 안에 넘어집니다. 그래서 검사용으로만 contact/fell 종료를 끄고 눕혀 두면, 첫 창(step 1–251)은 넘어지는 3-D 변위(약 0.65 m)를 이동으로 세고, **step 501에 목표가 1 m보다 먼 54개 env 전부에서 정확히 발동**. 종료 step의 패널티 = −500 × 0.02 = **−10**.
- 마지막 4 s 전에는 goal_position·goal_heading 합이 0.

## 7. 학습 명령

실행 중 (2026-11-21 02:56:49 PST 시작, pid 1214974, 시작 전 GPU에 다른 compute 프로세스 없음, RAM 54 GiB 여유):
```
nohup uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-Contact-FootholdD30-GoalReach-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name goal_reach_1024 --agent.logger tensorboard > ~/train_goal_reach_1024.log 2>&1 &
```
로그 디렉터리: `logs/rsl_rl/ame_g1/2026-11-21_02-56-55_goal_reach_1024`

Part B를 멈추고 Part A로 바꿔야 할 때:
```
nohup uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-AR025-Stag-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name ar025_stag_1024 --agent.logger tensorboard > ~/train_ar025_stag_1024.log 2>&1 &
```

## 8. 학습 시작 후 확인 (25 iteration, 03:01 PST)

- 로그: iteration 0 → 24 정상 증가, `nan`/Traceback 0건, iteration 9.7 s, GPU 약 9 GB / 11 GB.
- it 21→24: Mean reward −10.75 → −10.91, value loss 1.18 → 0.45, 에피소드 길이 50.9 → 53.8 step, action std 1.06.
- it 24 태그: goal/contact 0.63 (smoke 1.0에서 감소), goal/fell 0.44, goal/stagnation 0, goal/reached 0, final_distance 2.35 m, pelvis_speed 0.55, move_to_goal 0.107, termination_penalty −0.625, cone 10.4°, yaw_limit 0.039 rad. 시작 직후라 예상대로입니다.
- `params/env.yaml` 확인 (`logs/.../2026-11-21_02-56-55_goal_reach_1024/params/`):
  - `episode_length_s: 16.0`, decimation 4, `scale_rewards_by_dt: True`.
  - rewards: `goal_position` 100 (T 4, std 2), `goal_heading` 50 (T 2, dist 0.5), `move_to_goal` 5 (v 0.3–2.0, cos 0.5, dist 0.5), `stand_at_goal` 5 (dist 0.5, yaw 0.5), `termination_penalty` −500, `action_rate_l2` −0.025, `foothold` −1.0 δ 0.3; `track_linear_velocity`·`track_angular_velocity` 없음; pose/foot_clearance/foot_swing_height/foot_slip/soft_landing/air_time의 command는 `gait`; 나머지 정규화 항 weight는 base와 동일.
  - terminations: time_out, fell_over, out_of_terrain_bounds(time-out), illegal_contact(1 N), `stagnation` (`ame.mdp.goal.goal_stagnation`, 0.5 m / 1.0 m / 5 s, time_out False).
  - commands: `goal`, `gait`만 (twist 없음); goal: resample 1e9 s, 거리 [1.0, 3.5] (stairs [2.0, 3.5]), Chebyshev 3.5, heading ±π/3, cone 10°→45° / 48000 step.
  - curriculum: `ame.mdp.goal.terrain_levels_sgs_goal`, sgs {H 100, t 0.5, κ 1, T 1, eps 1e-8, uniform_mix 0.2}, reach 0.5 m, t_left < 4 s.
  - 관측: actor command = `goal_obs` dist_clip 2.0, time_left 없음; critic command = `goal_obs` clip 없음 + `time_left`.
  - reset_base `yaw_schedule` {steps 48000, limit π}.
  - `agent.yaml`: `DoubleCriticPPO`, reward group 2 = foothold, (1.0, 0.25), symmetry `ame.symmetry:mirror_obs_actions_goal`, seed 42, max_iterations 10000, run_name goal_reach_1024, logger tensorboard.

## 9. 아침에 확인할 것

TensorBoard: `logs/rsl_rl/ame_g1/2026-11-21_02-56-55_goal_reach_1024`. 약 10 s/it이므로 1k it ≈ 05:45, 2k it ≈ 08:30. `Episode_Reward/*`는 에피소드 합 ÷ 16 s이고, `Curriculum/terrain_levels/goal/*`는 최근 1000개 종료 에피소드 기준 비율입니다.

| 태그 | 시작값 | 1k–2k에서 정상 | 실패 신호 |
|---|---|---|---|
| `Curriculum/terrain_levels/goal/reached` | 0 | 꾸준히 증가, 2k에서 0.3 이상. SGS가 p≈0.5 레벨로 몰아주므로 0.4–0.6 근처 정체는 정상(쉬워진 만큼 어려운 레벨을 뽑음) | 1.5k 이후에도 < 0.1 |
| `.../goal/contact` | ≈1.0 | 빠르게 감소, 1k에서 0.5 이하 | 0.8 이상 유지 → 넘어지는 문제 |
| `.../goal/stagnation` | ≈0 (5 s 전에 넘어짐) | 넘어짐이 줄면서 잠깐 오를 수 있으나 2k에서 0.15 이하로 다시 감소 | 0.3 이상에서 계속 상승 → "멈춰 서기" attractor가 남음 (가장 중요한 실패 신호) |
| `.../goal/timeout` | 0 | reached와 함께 증가 | — |
| `.../goal/final_distance`, `Episode_Metrics/goal_distance` | ≈2.5 m | 1 m 아래로 감소 | 2 m 근처 정체 |
| `Episode_Metrics/pelvis_speed` | ≈0.3 (넘어지는 움직임) | 0.3–0.6 m/s (에피소드 평균, 도착 후 정지 포함) | 0.15 아래 → 정지 |
| `Episode_Reward/move_to_goal` | ≈0.02 | 상승, 최대 5.0 (매 step 1일 때) | — |
| `Episode_Reward/goal_position` | 0 | 상승, 최대 6.25 (마지막 4 s 동안 d=0) | 1k 이후에도 ≈0 |
| `Episode_Reward/goal_heading`, `stand_at_goal` | 0 | position보다 늦게 오름 | — |
| `Episode_Reward/termination_penalty` | −0.625 | −0.2 이상(덜 음수)으로 | −0.5 아래 유지 |
| `Episode_Reward/action_rate_l2` | ≈−0.02 | 작게 유지(가중치 −0.025) | 속도 보상보다 커짐 |
| `Mean episode length` (step) | ≈13 | 800 쪽으로 증가 | — |
| `Curriculum/terrain_levels/<family>/p_k`, `prob_k`, `<family>/reached` | 0, 0.1 | stepping_stones 낮은 레벨 p가 먼저 오르고 prob가 p≈0.5 레벨로 이동; stairs 높은 레벨 p가 0보다 커짐 | 모든 family에서 p가 0에 머묾 |
| `.../goal/cone_deg`, `goal/yaw_limit` | 10°, 0 | 2k it에서 45°, π에 도달 (선형) | — |
| loss / `Mean action std` | — | NaN 없음, std 서서히 감소 | NaN, value loss 폭주 |

판단: 2k it에서 `goal/reached`가 오르고 `goal/stagnation`이 낮으면 원래 문제(고정 보폭, 돌 앞 정지, 계단에서 제자리걸음)를 평가할 만한 정책입니다. 다음 확인은 계단 상위 레벨·stones 레벨별 `<family>/reached`와, play로 측보·보폭 변화를 보는 것입니다. 반대로 contact가 0.8 이상으로 남거나 stagnation이 오르기만 하면 Part A로 바꾸거나(§7 명령) termination penalty 대비 정규화 크기를 다시 보는 것을 권합니다. yaw/cone 커리큘럼이 2k it에서 끝나므로 그 직후에 reached가 잠깐 떨어지는 것은 정상입니다.
