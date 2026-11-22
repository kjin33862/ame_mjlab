notes/reward_audit/ — 71cf35f GoalReach report: note the push needs credentials

# Reward / stride audit (평가만, 정리 최소화한 원자료)

체크포인트: `logs/rsl_rl/ame_g1/2026-11-17_21-45-52_foothold_d30_alphanist_mix02_1024/model_8600.pt` (지정 경로 `~/ame_mjlab/checkpoints/foothold_d30_alphanist_mix02_1024/`는 존재하지 않음; 같은 run 이름의 로그 폴더에서 최고 번호 사용. 이 run의 env.yaml reward weight는 현재 태스크 cfg와 동일함을 확인).
태스크: Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1, actor만 로드. 조건당 16대, 10 s(500 step, dt 0.02), seed 42.

## 요약 (숫자만)
- Q1: 평지에서 걷기 total +7.06~+7.17/s. 정책이 스스로 서 있게 하고 vx1.0으로 채점(b3) +5.95/s → **서 있으면 약 1.2/s 낮음**. 차이는 거의 전부 track_linear_velocity(−1.68); 서 있을 때 pose(+0.18), foot_clearance(+0.10), action_rate(+0.07), track_angular(+0.07)로 일부 회수. 기본자세 고정(b1/b2)은 서 있지 못하고 계속 리셋돼 +2.1 / +4.5 (더 낮음).
- Q1(c): stones L2–3에서 정책의 멈춤 구간 +5.79/s vs 이동 구간 +5.67/s → **거의 같음 (멈춤이 +0.12/s 높음)**. 이동 쪽은 track_linear +0.81 이득이지만 track_angular(−0.23), foothold(−0.16), termination_penalty(−0.15), pose(−0.11), action_rate(−0.08), foot_clearance(−0.07)로 상쇄. 멈춤 비율: L2 21.6%, L3 11.5% of robot-time.
- Q2: 평지 step length는 vx에 따라 변함(0.064 / 0.188 / 0.226 m), **cadence는 vx와 무관하게 4.2–4.4 steps/s로 고정**, step width는 평지에서 거의 고정(0.22–0.24, std ≤ 0.008). 같은 vx1.0에서 지형에 따라 step length 분포가 바뀜(stones L1 0.214, L2 0.197, L3 0.161, stairs 오르막 0.130; 분산 크게 증가), width는 stones에서 좁아짐(0.15–0.18). action_rate/s는 평지에서 vx·step length와 거의 완전 선형(corr −0.99, ≈ −0.27×length − 0.022).

## 평가 설정 (학습 cfg 대비 바꾼 것, 모두 평가 전용)
- actor 관측 corruption off, height-map noise(bias/drift/jitter) off, push_robot·encoder_bias 제거, curriculum 제거. foot friction / COM 랜덤화(startup, seed 고정)와 reward·termination은 학습 그대로.
- 평가용 보드: 열 = flat(BoxFlat), stones L1/L2/L3(학습 보드 level L의 gap = 0.45·L/9 = 5/10/15 cm 고정), 역피라미드 계단 10..17 cm(1 cm 간격, 열당 2대). 행 4개 동일 → +x로 10 s 걸어도 같은 지형. spawn yaw 0, command는 로봇별 고정(vy=wz=0, heading/standing env 없음, resample 없음).
- reward: `reward_manager._step_reward`(= weight×raw, 초당 rate). 표의 값 = 10 s 전체 step 평균 = (step당 reward 합)/10 s. 리셋 포함 전 구간.
- (b) "기본 자세로 세움" = action 0 (JointPositionAction, use_default_offset → default pose).
- (c) 멈춤 = pelvis xy 속도 < 0.15 m/s가 1 s 넘게 연속(리셋에서 끊음), 나머지 = 이동.
- touchdown = `feet_ground_contact.compute_first_contact`이면서 직전 swing(`last_air_time`) ≥ 0.1 s. step length/width = 착지 발 ankle_roll_link − 반대 발, pelvis heading 좌표(앞, |옆|). spawn 후 0.5 s 제외. cadence = 로봇별 touchdown / 유효 초.
- 계단 "오르막" touchdown = 착지 발이 타일 중심에서 +x로 1.35–3.15 m(계단 구간, 바깥으로 갈수록 높음). 오르막 touchdown의 dz(착지발−지지발) 평균 +0.035 m로 확인.

## 측정 못 한 것 / 주의
1. **(b) 기본 자세 고정은 "서 있는 로봇"이 되지 않음.** default pose에서 손목(wrist_yaw_link)이 hip_roll_link에 닿음 → undesired_contact 센서가 secondary 필터 없이 자기충돌도 셈 → illegal_contact로 spawn 후 약 12 step(0.24 s)마다 리셋(b1, 16대 10 s에 558회). illegal_contact 종료만 끈 run(b2, `raw_noillegal.npz`)에서도 PD 기본 자세는 넘어짐(fell_over 106회). 그래서 (b3) "정책이 서 있게 함(관측 command만 0, reward는 vx1.0 기준)"을 추가함. "멈춤이 더 이득인가"의 실질 답은 b3와 (c).
2. contact 깜빡임: first-contact 이벤트의 55–66%가 air time ≈ 0.01 s (stance 중 chatter). 0.1 s 기준으로 제외함. 표의 괄호 숫자가 제외 전 개수.
3. 평지 vx0.5에서 실제 속도 0.32 m/s(명령의 64%), vx1.2에서 1.03. step length는 실제 속도 기준으로 해석해야 함.
4. stairs_17에서는 오르막 touchdown 11개, step length ≈ −0.01 → 17 cm 계단을 사실상 오르지 못함(제자리). 계단 높이별 표 참조.
5. 같은 seed라도 GPU(mujoco_warp) 비결정성으로 두 run 사이 종료 수가 1–2개 다름(stones L2 10 vs 11 등).
6. "step" = 한 발 착지 기준(반 stride). stride length는 약 2배.
7. 동영상 없음(요청 안 됨). Q1 그림은 b3−a(vx1.0), (c)멈춤−이동의 항목별 차이만 그림.

## 표

## Q1 — reward 항목별 초당 평균 (weight × raw, = 초당 reward)

| term | (a) flat vx0.5 | (a) flat vx1.0 | (a) flat vx1.2 | (b1) default pose, vx1.0 | (b2) default pose, vx1.0, no illegal-contact term. | (b3) policy stands (obs cmd 0), rewarded vs vx1.0 | (c) stones L2-3 moving | (c) stones L2-3 stopped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| track_linear_velocity | +1.690 | +1.721 | +1.690 | +0.130 | +0.487 | +0.038 | +0.858 | +0.047 |
| track_angular_velocity | +3.853 | +3.931 | +3.939 | +3.991 | +3.915 | +4.000 | +3.763 | +3.994 |
| upright | +0.990 | +0.983 | +0.980 | +0.928 | +0.654 | +0.994 | +0.949 | +0.983 |
| pose | +0.818 | +0.736 | +0.696 | +0.884 | +0.711 | +0.916 | +0.667 | +0.780 |
| body_ang_vel | -0.004 | -0.003 | -0.003 | -0.038 | -0.133 | -0.000 | -0.014 | -0.001 |
| angular_momentum | -0.009 | -0.013 | -0.014 | -0.065 | -0.355 | -0.000 | -0.030 | -0.002 |
| dof_pos_limits | +0.000 | +0.000 | -0.000 | -0.007 | -0.028 | +0.000 | -0.009 | +0.000 |
| action_rate_l2 | -0.040 | -0.071 | -0.085 | +0.000 | +0.000 | -0.001 | -0.088 | -0.004 |
| air_time | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| foot_clearance | -0.091 | -0.096 | -0.086 | -0.006 | -0.007 | -0.001 | -0.078 | -0.004 |
| foot_swing_height | -0.030 | -0.023 | -0.027 | -0.027 | -0.027 | -0.000 | -0.023 | -0.002 |
| foot_slip | -0.007 | -0.021 | -0.029 | -0.000 | -0.001 | -0.000 | -0.012 | -0.000 |
| soft_landing | -0.000 | -0.001 | -0.001 | -0.001 | -0.000 | -0.000 | -0.000 | -0.000 |
| self_collisions | +0.000 | +0.000 | +0.000 | -0.070 | -0.012 | +0.000 | -0.002 | +0.000 |
| undesired_contacts | +0.000 | +0.000 | +0.000 | -0.113 | -0.049 | +0.000 | -0.003 | +0.000 |
| termination_penalty | +0.000 | +0.000 | +0.000 | -3.487 | -0.663 | +0.000 | -0.150 | +0.000 |
| foothold | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | -0.157 | +0.000 |
| **total** | **+7.170** | **+7.143** | **+7.061** | **+2.118** | **+4.492** | **+5.945** | **+5.670** | **+5.792** |
| robot-seconds | 160 | 160 | 160 | 160 | 160 | 160 | 267 | 53 |
| terminations | 0 | 0 | 0 | 558 | 106 | 0 | – | – |
| mean pelvis xy speed (m/s) | 0.32 | 0.85 | 1.03 | 0.14 | 0.51 | 0.01 | 0.65 | 0.05 |

- stones_L2: stopped 21.6% of robot-time, robots with any stopped segment 9/16, terminations 10
- stones_L3: stopped 11.5% of robot-time, robots with any stopped segment 7/16, terminations 30

## Q2 — touchdown 통계 (mean ± std [p10, p90])

| 조건 | touchdown 수 (chatter 제외 전 first-contact 수) | step length (m, heading 앞방향) | step width (m, 좌우 거리) | cadence (steps/s, 로봇별) | action_rate (초당, 로봇별) | pelvis 속도 (m/s) |
|---|---:|---|---|---|---|---:|
| flat vx0.5 | 669 (1446) | 0.064 ± 0.020 [0.049, 0.075] | 0.237 ± 0.005 [0.234, 0.241] | 4.392 ± 0.140 [4.202, 4.569] | -0.040 ± 0.004 [-0.045, -0.036] | 0.32 |
| flat vx1.0 | 640 (1426) | 0.188 ± 0.013 [0.177, 0.197] | 0.226 ± 0.007 [0.223, 0.231] | 4.202 ± 0.111 [4.097, 4.307] | -0.071 ± 0.002 [-0.072, -0.068] | 0.85 |
| flat vx1.2 | 640 (1527) | 0.226 ± 0.014 [0.216, 0.236] | 0.222 ± 0.008 [0.217, 0.227] | 4.202 ± 0.117 [4.044, 4.307] | -0.085 ± 0.002 [-0.088, -0.083] | 1.03 |
| stones L1 vx1.0 | 586 (1290) | 0.214 ± 0.044 [0.167, 0.245] | 0.175 ± 0.023 [0.151, 0.209] | 3.847 ± 0.082 [3.782, 3.939] | -0.091 ± 0.002 [-0.094, -0.089] | 0.93 |
| stones L2 vx1.0 | 336 (991) | 0.197 ± 0.087 [0.083, 0.268] | 0.159 ± 0.041 [0.112, 0.219] | 2.307 ± 0.676 [1.681, 3.137] | -0.060 ± 0.020 [-0.088, -0.040] | 0.58 |
| stones L3 vx1.0 | 332 (1064) | 0.161 ± 0.129 [-0.023, 0.303] | 0.149 ± 0.064 [0.075, 0.226] | 2.413 ± 0.349 [1.995, 2.843] | -0.087 ± 0.026 [-0.128, -0.060] | 0.53 |
| stairs up 10–17 cm vx1.0 (오르막 구간 touchdown) | 165 (–) | 0.130 ± 0.089 [0.002, 0.208] | 0.220 ± 0.036 [0.174, 0.263] | 3.649 ± 0.885 [2.101, 4.202] | -0.090 ± 0.027 [-0.114, -0.043] | 0.63 |
| stairs 전체 (평지·오르막·내리막) | 554 (–) | 0.145 ± 0.071 [0.037, 0.215] | 0.229 ± 0.028 [0.192, 0.262] | 3.649 ± 0.885 [2.101, 4.202] | -0.090 ± 0.027 [-0.114, -0.043] | 0.63 |
| (b3) policy stands | 0 (32) | – | – | 0.000 ± 0.000 [0.000, 0.000] | -0.001 ± 0.000 [-0.001, -0.001] | 0.01 |

- stairs 오르막 touchdown의 발 높이 차(착지발 − 지지발) dz: 0.035 ± 0.069 [-0.068, 0.113] m (오르막 확인용)
- stairs 평지(웅덩이 바닥·윗면): n=280, length 0.167 ± 0.054 [0.103, 0.231], width 0.231 ± 0.020 [0.207, 0.257]
- stairs 오르막: n=165, length 0.130 ± 0.089 [0.002, 0.208], width 0.220 ± 0.036 [0.174, 0.263]
- stairs 내리막: n=109, length 0.109 ± 0.060 [0.050, 0.184], width 0.240 ± 0.024 [0.218, 0.267]

| 계단 높이 | 오르막 touchdown | step length | step width |
|---:|---:|---|---|
| stairs_10 | 17 | 0.176 ± 0.050 [0.132, 0.219] | 0.219 ± 0.021 [0.195, 0.238] |
| stairs_11 | 21 | 0.165 ± 0.049 [0.114, 0.202] | 0.225 ± 0.023 [0.196, 0.255] |
| stairs_12 | 23 | 0.139 ± 0.077 [0.018, 0.205] | 0.204 ± 0.028 [0.180, 0.242] |
| stairs_13 | 19 | 0.164 ± 0.053 [0.120, 0.207] | 0.202 ± 0.026 [0.179, 0.245] |
| stairs_14 | 23 | 0.146 ± 0.075 [0.043, 0.202] | 0.216 ± 0.048 [0.171, 0.266] |
| stairs_15 | 32 | 0.119 ± 0.109 [0.007, 0.210] | 0.222 ± 0.041 [0.161, 0.265] |
| stairs_16 | 19 | 0.084 ± 0.086 [-0.004, 0.193] | 0.231 ± 0.040 [0.174, 0.279] |
| stairs_17 | 11 | -0.010 ± 0.035 [-0.057, 0.016] | 0.255 ± 0.014 [0.231, 0.273] |

action_rate 관계 (보행 조건 로봇 128대 기준, b3 제외; 로봇 1대 = 점 1개):
- corr(action_rate/s, step length) = -0.64
- corr(action_rate/s, cadence) = +0.02
- flat만 (vx 0.5/1.0/1.2, 48대): corr(action_rate/s, vx) = -0.99, corr(action_rate/s, step length) = -0.99
- flat 선형근사: action_rate/s ≈ -0.270 × step length -0.022

## 그림
- `q1_reward_diff.png`: 항목별 초당 reward 차이 (왼쪽 b3 − a vx1.0, 오른쪽 stones L2–3 멈춤 − 이동).
- `q2_stride.png`: 조건별 step length / step width 분포(상자 p25–p75, 수염 p10–p90), 로봇별 action_rate vs step length.

## 파일
`reward_audit.py`(측정), `analyze.py`(표·그림), `raw.npz`, `raw_noillegal.npz`(원자료), `tables.md`. 커밋 안 함.
