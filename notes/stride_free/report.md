구현 커밋: `4775041` (브랜치 `stride-free`, 부모 `7135e11`) · 이 보고서는 그 다음 커밋에 들어 있다.

# 보폭 자유화(StrideFree) 과제 준비 보고 (학습 안 함)

## 결론

1. **새 과제 `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-StrideFree-Velocity-Rough-G1` 등록.**
   FootholdD30 과제(7135e11)와 다른 점은 두 보상 항뿐이다. 학습·play env 설정 diff는 이 두 항의 줄뿐이고, rl 설정 diff는 0줄이다(§3).
   - a. `track_linear_velocity`: 순간 몸통(골반) 속도 대신, **골반 xy 속도(yaw 기준)를 시간상수 1.0 s로 지수 이동평균한 값**을 명령과 비교한다.
     가중치 2.0, std 0.5는 그대로다. 리셋 직후에는 평균을 현재 속도로 다시 시작한다.
   - b. `pose`(variable_posture)의 걷기 std에서 **hip_pitch 0.3→0.6, hip_roll 0.15→0.3, knee 0.35→0.7**만 바꿨다. 다른 관절과 서 있기·달리기 std는 그대로다.
2. **기존 과제는 바뀌지 않는다.** 두 변경 모두 `ame_gap_env_cfg`의 설정 값(`lin_vel_ema_tau`, `walking_posture_std_scale`)이고, 기본값 `None`이면 기존 mjlab 항을 그대로 쓴다.
   7135e11과 현재 브랜치에서 기존 과제 29개의 env(train)·env(play)·rl 설정 repr 해시를 비교했고, 모두 같았다.
3. **검증:** 테스트 120개 통과, 1개 skip(onnxruntime 미설치, 기존 skip). 이 중 10개가 새 테스트다.
   64 env × 3 iteration 스모크 학습은 정상 종료했다(exit 0).
   실제 env 8개로 400 step을 돌리며 확인한 결과, 항 내부의 이동평균은 직접 계산한 값과 최대 1.9e-9 차이였다(리셋 후 재시작 229회 포함).
4. **이어서 학습할 명령(2080 Ti)은 §5에 있다.**
   `--agent.resume True --agent.load-run '.*_foothold_d30_alphanist_mix02_1024$' --agent.load-checkpoint 'model_9999\.pt$' --agent.max-iterations 4000`.
   iteration 번호는 9999부터 13998까지다.
5. **확인이 필요한 점 세 가지(§6):**
   - (i) 새 걷기 std가 달리기 std보다 커진다(예: knee 걷기 0.7 > 달리기 0.6). 명령 속도 합이 1.5 이상이면 오히려 더 엄격해진다.
   - (ii) 이어서 학습하면 SGS 성공 버퍼는 checkpoint에 저장되지 않아 비어 있는 상태로 시작한다.
   - (iii) 명령이 바뀐 뒤 약 1 s 동안은 이동평균이 늦게 따라와서, 추종 보상이 정책과 관계없이 낮다.

## 1. 현재 과제(FootholdD30 AlphaNIST)의 보상 항 전체

과제 `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1`, 학습 설정을 `load_env_cfg`로 읽었다
(`notes/stride_free/reward_table.py`). 가중치는 mjlab 보상 관리자에서 step_dt(0.02 s)가 곱해지기 전 값이다.
double critic에서 `foothold`만 두 번째 보상 그룹이고(A = 1.0 A1 + 0.25 A2), 나머지는 첫 번째 그룹이다.

| 항 | 가중치 | 함수 | std / 주요 파라미터 |
|---|---:|---|---|
| `track_linear_velocity` | 2.0 | mjlab `track_linear_velocity` (골반, 몸체 좌표, xy + z² 오차) | std 0.5 (√0.25) |
| `track_angular_velocity` | 4.0 | `ame.mdp.rewards.track_ang_vel_z_world_exp` (골반, 월드 yaw rate만) | std 0.7071 (√0.5) |
| `upright` | 1.0 | mjlab `upright` (torso_link) | std 0.4472 (√0.2) |
| `pose` | 1.0 | mjlab `variable_posture` (모든 관절) | 아래 표, walking_threshold 0.05, running_threshold 1.5 |
| `body_ang_vel` | −0.05 | mjlab `body_angular_velocity_penalty` (torso_link) | – |
| `angular_momentum` | −0.02 | mjlab `angular_momentum_penalty` | sensor `robot/root_angmom` |
| `dof_pos_limits` | −1.0 | mjlab `joint_pos_limits` | – |
| `action_rate_l2` | −0.1 | mjlab `action_rate_l2` | – |
| `air_time` | 0.0 (꺼짐) | mjlab `feet_air_time` | threshold 0.05–0.5 s, command_threshold 0.5 |
| `foot_clearance` | −2.0 | `ame.mdp.swing.feet_clearance_one_sided` | target 0.10 m, command_threshold 0.05 |
| `foot_swing_height` | −0.25 | `ame.mdp.swing.feet_swing_height_one_sided` | target 0.10 m, command_threshold 0.05 |
| `foot_slip` | −0.1 | mjlab `feet_slip` | command_threshold 0.05 |
| `soft_landing` | −1e-5 | mjlab `soft_landing` | command_threshold 0.05 |
| `self_collisions` | −1.0 | mjlab `self_collision_cost` | force_threshold 10 |
| `undesired_contacts` | −1.0 | `ame.mdp.rewards.undesired_contacts` | threshold 1.0 |
| `termination_penalty` | −50.0 | mjlab `is_terminated` | – |
| `foothold` | −1.0 | `ame.mdp.foothold.foothold_unsupported` (보상 그룹 2) | δ 0.30 m |

`pose`의 관절별 std(standing은 전 관절 0.05). **굵은 글씨**가 이번에 바뀐 값이다.

| 관절 패턴 | walking (기존 → StrideFree) | running |
|---|---|---|
| hip_pitch | **0.3 → 0.6** | 0.5 |
| hip_roll | **0.15 → 0.3** | 0.2 |
| hip_yaw | 0.15 | 0.2 |
| knee | **0.35 → 0.7** | 0.6 |
| ankle_pitch | 0.25 | 0.35 |
| ankle_roll | 0.1 | 0.15 |
| waist_yaw / roll / pitch | 0.2 / 0.08 / 0.1 | 0.3 / 0.08 / 0.2 |
| shoulder_pitch / roll / yaw | 0.15 / 0.15 / 0.1 | 0.5 / 0.2 / 0.15 |
| elbow | 0.15 | 0.35 |
| wrist | 0.3 | 0.3 |

regime은 명령 속도 합 |v_xy| + |ω_z|로 고른다. 0.05 미만은 standing, 0.05 이상 1.5 미만은 walking, 1.5 이상은 running이다.

## 2. 변경 내용

### a. 이동평균 선속도 추종 (`ame.mdp.rewards.track_lin_vel_xy_ema_exp`)

```
v_yaw  = R_yaw(골반)^T · v_world(골반)            # xy만
v_ema ← v_ema + α (v_yaw − v_ema),  α = 1 − exp(−step_dt / τ) = 0.019801  (τ = 1.0 s, step_dt 0.02 s)
r = exp(−(|cmd_xy − v_ema|² + v_body,z²) / std²),  std = 0.5, 가중치 2.0
```

- **몸체:** 기존 항과 같은 골반(root link)을 쓴다. 이 저장소의 AME 스택은 torso_link를 쓰지만, 이 과제의 기존 항이 골반 기준이라 그대로 두었다.
- **z 오차:** mjlab 항의 z² 오차(몸체 좌표 z 속도, 순간값)는 그대로 남겼다. xy만 바꾸라는 지시에 맞춰 범위를 최소로 했다.
- **리셋:** 리셋할 때 평균을 현재 속도로 초기화한다. 다만 mjlab은 `_reset_idx` 안에서 보상 항의 `reset()`을 부르는데, 이 시점은 `sim.forward()` 전이라
  속도 값(cvel)이 아직 리셋 전 상태다. 그래서 `reset()`에서는 해당 env에 표시만 해 둔다. 다음 보상 계산 때 그 시점 속도로 평균을 다시 시작하고,
  그 값으로 첫 보상을 낸다. mjlab 보상은 모두 한 physics substep 늦은 값을 보는데(mjlab 설계), 이 항도 똑같은 시점 값을 쓴다.
- **효과(단위 테스트로 확인):** 1 Hz 보폭에 ±0.3 m/s 전후·좌우 흔들림이 있을 때, 평균이 안정된 뒤 이동평균 항은 최소 0.97을 준다.
  같은 조건에서 mjlab 순간 항의 평균은 0.75 미만이다. 즉 보폭 안에서의 속도 변화에는 더 이상 벌점이 거의 없다.

### b. 걷기 자세 std (`STRIDE_FREE_WALKING_STD_SCALE`)

`{".*hip_pitch.*": 2, ".*hip_roll.*": 2, ".*knee.*": 2}`를 기존 `std_walking` 값에 곱한다.
G1 관절 이름으로 풀어 보면 바뀌는 관절은 left/right × hip_pitch, hip_roll, knee의 정확히 6개다(테스트로 확인). 없는 패턴 키를 주면 ValueError가 난다.

### 설정 값과 파일

- `src/ame/env_cfg.py`: `ame_gap_env_cfg(..., lin_vel_ema_tau=None, walking_posture_std_scale=None)`, `apply_lin_vel_ema`, `apply_walking_posture_std_scale`,
  `STRIDE_FREE_WALKING_STD_SCALE`. 각 mutator는 대상 항이 mjlab 원래 함수인지 확인하고, 아니면 ValueError를 낸다.
- `src/ame/mdp/rewards.py`: `track_lin_vel_xy_ema_exp`, `ema_alpha`, `LIN_VEL_EMA_TAU`.
- `src/ame/__init__.py`: `STRIDE_FREE_FLAGS`, 새 과제 등록(train·play 모두 a, b 적용).
- `tests/test_stride_free.py` (10개): α 값, 리셋 직후 mjlab 항과 같은 값, 갱신식과 env별 리셋, 시간상수(1 s 후 1 − 1/e),
  보폭 흔들림 평균화, yaw 좌표 xy와 몸체 좌표 z, 기본값 꺼짐, std 배율과 잘못된 키, 6개 관절만 바뀜, D30 과제와의 차이.

## 3. 설정 diff (FootholdD30 → StrideFree, `config_diff.txt`)

```
=== env (train): 29 changed lines
-rewards={'track_linear_velocity': RewardTermCfg(func=<function track_linear_velocity>,
+rewards={'track_linear_velocity': RewardTermCfg(func=<class 'ame.mdp.rewards.track_lin_vel_xy_ema_exp'>,
-'std': 0.5},
+'std': 0.5,
+'tau': 1.0,
+'asset_cfg': SceneEntityCfg(name='robot', ...기본값 22줄...)},
-'std_walking': {'.*hip_pitch.*': 0.3, '.*hip_roll.*': 0.15, ..., '.*knee.*': 0.35, ...},
+'std_walking': {'.*hip_pitch.*': 0.6, '.*hip_roll.*': 0.3, ..., '.*knee.*': 0.7, ...},
=== env (play): 29 changed lines   (같은 두 항)
=== rl: 0 changed lines
```

`asset_cfg`는 기본값(`SceneEntityCfg('robot')`, 즉 root link)이고, mjlab 항이 인자를 생략했을 때 쓰는 값과 같다. a, b 외 차이는 없다.
δ 0.30, AlphaNIST + uniform_mix 0.2, 종료 조건, 지형, seed(rl 설정의 42), double critic 설정은 모두 같다.
기존 과제 해시: `config_hashes_7135e11.txt`(29줄)와 `config_hashes_new.txt`(30줄)에서, 새 과제 한 줄을 빼면 diff가 없다.

## 4. 테스트와 스모크

- `pytest tests`: **120 passed, 1 skipped** (7135e11 시점 110 + 1 skip, 새 테스트 10개).
- 스모크: 이 노트북(RTX 5060 Laptop)에서
  `train <새 과제> --env.scene.num-envs 64 --agent.max-iterations 3 --agent.seed 42 --agent.logger tensorboard`를 돌렸다.
  exit 0, iteration당 1.4 s, 보상표에 `track_linear_velocity` 2.0이 들어간 것을 확인했다.
- 실제 env 확인(`ema_runtime_check.py`/`.txt`): env 8개, episode 2 s, 무작위 행동 400 step. 보상 관리자가 보는 골반 속도로 이동평균을 따로 계산했고,
  항 내부 값과의 최대 차이는 1.86e-9였다. 리셋 표시는 매 step 정확히 리셋된 env와 같았고, 리셋 후 재시작은 229회였다.

## 5. 2080 Ti에서 이어서 학습

이어서 학습 옵션은 mjlab `RslRlBaseRunnerCfg`(`mjlab/rl/config.py`)에 있다.

- `resume: bool`. CLI에서는 `--agent.resume True`로 준다(tyro bool 필드라 값을 꼭 써야 한다).
- `load_run: str`. `logs/rsl_rl/<experiment_name>/` 아래 run 디렉터리를 고르는 정규식이고, 맞는 것 중 이름순 마지막을 쓴다.
- `load_checkpoint: str`. checkpoint 파일 이름 정규식이다.

`mjlab/scripts/train.py`가 `get_checkpoint_path(log_root/experiment_name, load_run, load_checkpoint)`로 경로를 찾은 뒤 `runner.load()`를 부른다.
`runner.load()`는 actor·critic·optimizer를 불러오고, `iter`와 env의 `common_step_counter`도 되살린다.
rsl_rl `learn(num_learning_iterations=max_iterations)`은 `current_learning_iteration`부터 그 수만큼 돈다.
따라서 `--agent.max-iterations 4000`이면 iteration 9999–13998이고, 마지막 저장 파일은 `model_13998.pt`이다.
experiment_name은 두 과제 모두 `ame_g1`이라(rl 설정 동일), 이전 run이 같은 디렉터리에 있어야 한다.

```bash
cd ~/ame_mjlab_sgs                 # 2080 Ti 쪽 저장소 경로에 맞게
git fetch mine
git checkout mine/stride-free      # detached HEAD
ls logs/rsl_rl/ame_g1/ | grep foothold_d30_alphanist_mix02_1024    # 이전 run이 정확히 하나인지, model_9999.pt가 있는지 확인
mkdir -p logs
nohup uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-StrideFree-Velocity-Rough-G1 \
  --env.scene.num-envs 1024 --agent.max-iterations 4000 --agent.seed 42 \
  --agent.run-name stride_free_ft_1024 --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run '.*_foothold_d30_alphanist_mix02_1024$' \
  --agent.load-checkpoint 'model_9999\.pt$' \
  > logs/stride_free_ft_1024.log 2>&1 &
grep -m1 "Loading model checkpoint" logs/stride_free_ft_1024.log   # 불러온 경로 확인
```

정규식은 mjlab `get_checkpoint_path`에 가짜 디렉터리를 넣어 확인했다. 대상 run의 `model_9999.pt`를 고르고(`model_10000.pt`는 고르지 않음),
새 run(`..._stride_free_ft_1024`)과 이름 뒤에 접미사가 붙은 run은 맞지 않는다.

## 6. 주의점

1. **walking std가 running보다 커진다.** hip_pitch 0.6 > 0.5, hip_roll 0.3 > 0.2, knee 0.7 > 0.6이다.
   그래서 명령 속도 합이 1.5 이상(예: vx 1.2 + |ωz| 0.4)이면 오히려 더 엄격한 std가 적용된다. 지시대로 walking만 바꿨다.
   일관성을 원하면 running도 같은 관절을 최소 걷기 값까지 올려야 한다(예: ×1.4). 그 경우 설정 값 하나만 더 추가하면 된다.
2. **SGS 버퍼는 checkpoint에 없다.** 이어서 학습을 시작하면 레벨별 성공 버퍼(H = 100)가 비어 p_k = 0이므로, 처음에는 레벨이 균등하게 뽑힌다.
   버퍼가 다시 찰 때까지 어려운 레벨(징검다리 5–9 등) 표본이 평소보다 많다.
3. **명령 전환 지연.** 명령은 3–8 s마다 다시 뽑힌다. τ = 1 s이므로 전환 후 1 s 동안 이동평균은 새 명령까지의 63%만 따라가고,
   이 구간에서는 잘 추종해도 보상이 낮다. 정지 명령(5%)이나 방향 전환 직후에는 실제보다 낮게 평가된다.
4. **critic 불일치.** 불러온 critic은 기존 보상(순간 속도, 좁은 std)으로 학습됐다. 초반 몇백 iteration은 가치 추정이 다시 맞춰지면서
   value loss가 튈 수 있다.
5. 학습은 하지 않았다. 보폭과 옆걸음 폭이 실제로 넓어지는지는 학습 후 평가해야 알 수 있다.
