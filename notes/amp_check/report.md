# AMP check: AMP 구현과 태스크 2개

## 결론

- AMP를 `src/ame/amp.py`에 다시 구현했고 태스크 2개를 등록했습니다 (branch `amp`, 기반 `foothold-v2` 034c4e7).
  - **태스크 1** `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-AMP-Velocity-Rough-G1`: FootholdV2-Progress와 env·play config의 hash가 완전히 같고, 다른 것은 algorithm(AMP)뿐입니다.
  - **태스크 2** `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-AME2Contact-FootholdD30-AMP-Velocity-Rough-G1`: D30에서 `illegal_contact` 종료를 빼고, AME-2의 torso/pelvis 400 N 종료를 넣고, AMP를 붙였습니다. 나머지(보상, 다른 종료, 센서)는 같습니다.
- 모든 검사를 통과했습니다.
  - 기존 33개 태스크의 config hash는 그대로이고, 새 태스크 2개만 추가됐습니다.
  - 데이터 replay에서 관절·body 순서가 일치합니다.
  - AMP 관측의 layout은 데이터 쪽과 로봇 쪽이 일치합니다 (최대 오차 1.3e-5).
  - smoke run 2개 모두 오류와 NaN이 없고, D(expert) > D(policy)입니다 (iteration 0부터).
  - 태스크 2에서 무릎 접촉은 episode를 끝내지 않고 `undesired_contacts`로 감점됩니다.
  - pytest 163 passed, 1 skipped입니다.
- 속도: 태스크 1은 10.36 s/it입니다 (FootholdV2는 9.8 s/it). 10000 it에 **약 29시간** 걸립니다.
- 가장 큰 우려는 style reward의 크기입니다. 기본값(coef 0.1, lerp 0.75)에서 smoke 중 style은 step당 약 0.013, group 1에 들어가는 몫은 0.25 × 0.013 ≈ 0.003입니다. task 쪽 group 1 return은 약 −0.76입니다. hopping을 바꿀 만큼 강한지는 학습해 봐야 압니다 (§5).

## 1. Reference 사실 (AMP_mjlab @ 6c7a294, `~/ref/AMP_mjlab`, 읽기 전용)

| 항목 | AMP_mjlab | 우리 |
|---|---|---|
| mjlab | 1.2.0 (`setup.py`) | 1.4.0 |
| `mjlab_patch` | observation_manager에 `history_ordering="time"` 추가 (actor/critic history 4의 시간순 배치용, AMP와 무관) | 필요 없음. `csv_to_npz.py`도 우리 mjlab 1.4.0에서 수정 없이 동작 |
| rsl_rl | 자체 fork (legacy `ActorCritic`, `AMPPPO`, `AmpOnPolicyRunner`) | rsl_rl 5.4.0. 그래서 다시 구현함 |
| 관절 순서 | CSV 29개 관절 = `csv_to_npz.py`의 `joint_names` | mjlab `g1.xml` 관절 순서와 같음 (확인함) |
| body 이름·순서 | tracking G1 scene의 30 body | 우리 env의 robot 30 body와 같음 (확인함) |
| AMP 관측 | 13 body: pelvis, L/R hip_roll, knee, ankle_roll, L/R shoulder_roll, elbow, wrist_yaw | 같음 |
| 기준 frame | anchor `torso_link`. pos와 rotation matrix 앞 2열은 anchor 기준, lin/ang vel은 각 body 자신의 frame | 같음 |
| 차원 | 한 frame 195 = 13 × (3+6+3+3), 순서 [pos \| ori \| lin_vel \| ang_vel]. 한 transition은 연속 2 frame이고 D 입력은 390 | 같음 |
| fps | CSV 120 fps를 `csv_to_npz.py`로 50 Hz로 리샘플 (pos·관절 lerp, quat slerp). 속도는 `torch.gradient`와 so3 미분을 sim.forward에 넣어 얻은 body 속도 | 정책 dt 0.02 s = 50 Hz라 그대로 씀 |
| style reward | `r_s = 0.1 · max(0, 1 − ¼(D−1)²)` (`discriminator.py:predict_amp_reward`) | 같음 |
| 섞는 방식 | `r = (1−0.75)·r_s + 0.75·r_task`. **lerp 0.75는 task reward 쪽** | group 1에 적용: `r1 ← 0.25·r_s + 0.75·r1`. group 2는 그대로 (자체 std로 정규화되므로 0.75를 곱해도 advantage는 같음) |
| 종료 transition | mjlab이 reset 후에 관측하므로 next = 직전 frame | 같음 |
| Discriminator | MLP 1024-512-256 ReLU + linear 1 | 같음 |
| D loss | `0.5·(MSE(D_e, 1) + MSE(D_p, −1))` | 같음 |
| gradient penalty | `10 · ‖∂D/∂x‖²`. 정규화하지 않은 expert 입력에서 계산함 (reference가 그렇게 함) | 그대로 재현 |
| 입력 정규화 | running mean/var (count 초기값 1e-4, eps 1e-4, ±10 clip). 각 step 뒤에 policy·expert state로 갱신 | 같음 |
| 학습률·optimizer | PPO와 같은 Adam, 같은 adaptive lr. weight decay: trunk 1e-3, head 1e-1. gradient clip은 policy에만 | 같음 (param group 추가) |
| D update 횟수 | PPO minibatch마다 1회 = epochs × minibatches (5 × 4 = 20) | 5 × 16 = 80 (우리 minibatch 수 16을 따름. 아래 우려 2) |
| D batch | num_envs × 24 / minibatches | 1024 envs에서 1536 |
| expert 샘플링 | minibatch마다 clip 하나 (정렬 순서로 순환), frame uniform, next = min(i+1, T−1) | 같음 |
| replay buffer | 100000 transition, uniform 복원 추출 | 같음 |
| 그 밖의 설정 | `min_normalized_std` 0.05, entropy 0.005, obs normalization, history 4 | AMP와 무관해서 가져오지 않음 |

**라이선스.** 저장소 최상위에 LICENSE가 없습니다. vendored `rsl_rl` 파일 헤더에는 BSD-3-Clause(RSL-RL / Legged Lab / TienKung-Lab)가 적혀 있습니다. 코드는 복사하지 않고 docstring에 출처를 적어 다시 썼습니다. 모션 데이터는 저장소 밖에서 읽습니다 (`~/ref/amp_mjlab_npz/amp`, 환경변수 `AME_AMP_MOTION_DIR`로 바꿀 수 있음). clip의 원 출처와 라이선스는 저장소에 명시돼 있지 않습니다.

**AME-2 접촉 규칙** (ame2_minimal @ 34b6fe9, `~/ref/ame2_minimal`):
- 종료 `early_illegal_contact`: `torso_link`, `pelvis_contour_link`의 접촉력 norm을 history 4 (= 4 substep × 0.005 s, 정책 1 step) 동안 평균한 값이 **400 N** 초과일 때. episode 시작 **1.0 s 이후**에만 적용 (`mdp/terminations.py:81-111`, `ame2_env_cfg.py:597-603`). 400은 AME-2 cfg의 `robot_weight`입니다 (`ame2_env_cfg.py:529`). 우리 G1의 실제 무게는 33.3 kg ≈ 327 N입니다.
- 감점 `undesired_non_foot_contacts` (weight −1): ankle·카메라 link를 뺀 `*_link` 중 4 substep 최대 접촉력이 1 N을 넘는 개수, 여기에 그 step에 처음 닿은 link 수를 한 번 더 더함 (`mdp/rewards.py:399-413`).
- 선택: 우리 `undesired_contacts` (weight −1, 비발 link 7종의 현재 net force > 1 N 개수)를 그대로 썼습니다. 차이(wrist 포함, substep 최대값, 첫 접촉 추가 계수)는 작아서 "clearly differs"가 아니라고 판단했습니다. 우리 G1의 pelvis 충돌 geom은 `pelvis` body에 있어 `pelvis_contour_link` 대신 `pelvis`를 씁니다.

## 2. 구현 (file:line)

| 내용 | 위치 |
|---|---|
| AMP 관측 함수 (데이터와 로봇이 같은 함수를 씀) | `src/ame/amp.py:117` |
| clip 17개 목록 (walk 8, jog 6, idle 3) | `src/ame/amp.py:82` |
| expert 데이터 로더와 샘플러 | `src/ame/amp.py:145`, `:185` |
| Discriminator, gradient penalty | `src/ame/amp.py:196`, `:211` |
| 정규화기, replay buffer | `src/ame/amp.py:218`, `:244` |
| `AmpDoubleCriticPPO` (D param group은 PPO Adam에 추가) | `src/ame/amp.py:265`, `:296-301` |
| 로봇 AMP 관측, style reward, group 1에 섞기 | `src/ame/amp.py:324`, `:338`, `:345`, `:357` |
| D 학습 (PPO minibatch마다 optimizer.step 직전) | `src/ame/amp.py:364`, `:376` |
| checkpoint에 D와 정규화기 저장·로드 | `src/ame/amp.py:427`, `:433` |
| DoubleCriticPPO의 group 분리 hook (동작은 같음) | `src/ame/double_critic.py:183` |
| runner cfg | `src/ame/rl_cfg.py:223`, `:237`, `:246`, `:251` |
| torso 종료 함수 | `src/ame/mdp/terminations.py:27` |
| torso 센서 (history = decimation 4), 종료 적용 | `src/ame/reward_cfg.py:281-286`, `:289`, `:304` |
| 태스크 2 env | `src/ame/env_cfg.py:620` |
| 등록 | `src/ame/__init__.py:480`, `:491` |
| 테스트 10개 | `tests/test_amp.py` |

로깅 (tensorboard `Loss/`): `amp_loss`, `amp_grad_pen`, `amp_d_expert`, `amp_d_policy` (update 평균), `amp_style_reward`, `amp_d_policy_rollout` (rollout 평균).

## 3. 검사 결과

| 검사 | 결과 |
|---|---|
| 기존 태스크 불변 | 33개 모두 hash 동일. 추가 2개 (`config_hashes_{old,new,diff}.txt`, `config_hashes.py`) |
| 태스크 1 = FootholdV2 + AMP | env·play hash 같음. rl cfg 차이는 algorithm class와 `amp_*` 6개 필드뿐 |
| 태스크 2 = D30 + 접촉 규칙 + AMP | env 차이는 센서 `torso_contact` 추가, `illegal_contact` 제거, `torso_collision` 추가뿐. 보상 동일 |
| 데이터 변환 | `csv_to_npz.py`로 17개 변환, 50 Hz, 30 body (`~/ref/amp_mjlab_npz/amp`) |
| 관절·body 순서 | CSV 관절 순서 = g1.xml 순서. npz joint_pos로 FK한 body 위치와 npz body 위치의 차이 ≤ 2e-6 m. frame 0 관절과 CSV(이름 매핑)의 차이 ≤ 6e-8 rad. 관절 범위 밖 값 0 (`replay_check.py`) |
| 발 높이 (sole 최저점) | 걷기·뛰기 clip은 최소 −0.035 ~ −0.008 m (바닥 아래로 들어감), 최대 0.05–0.37 m. 그림 `walk_forward_loop_002__A022_frames.png`, `jog_forward_loop_003__A021_frames.png` (측면, 자연스러운 무릎 굽힘) |
| AMP 관측 layout | task 1 env에서 로봇을 clip 상태로 둔 뒤 계산한 로봇 쪽 관측과 데이터 쪽 관측: 8개 clip 모두 (8, 195) vs (195,). 최대 차이 1.3e-5 (pos·ori·lin·ang 블록 모두) (`shape_check.py`) |
| smoke 태스크 1 (1024 envs, 30 it) | exit 0, NaN 없음. D_e/D_p: it 0 0.52/−0.61, it 5 0.88/−0.85, it 29 0.88/−0.88. amp_loss 0.29 → 0.04, grad_pen 0.17 → 0.08, style 0.066 → 0.013 (유한). **10.36 s/it** (수집 1.8 s, 학습 8.5 s) |
| smoke 태스크 2 (1024 envs, 10 it) | exit 0, NaN 없음. D_e/D_p: it 0 0.47/−0.63, it 9 0.86/−0.89. episode 길이 약 58 step (태스크 1은 약 5). 종료는 fell_over와 torso_collision. undesired_contacts −0.01 수준. 10.4 s/it |
| 무릎 접촉 (`knee_probe.py`, 64 envs, 무작위 행동, 400 step) | 태스크 2: 무릎 > 1 N인 env-step 468개 모두 `undesired_contacts` ≤ −1. 그중 420개는 계속 진행했고, 끝난 48개는 fell_over 46, torso_collision 2 (무릎 자체로 끝난 것은 없음). D30: 무릎 접촉 3개 모두 `illegal_contact`로 종료 |
| checkpoint | `discriminator_state_dict`, `amp_normalizer_state_dict` 포함. optimizer param group 3개 (wd 0, 1e-3, 0.1) |
| pytest | 163 passed, 1 skipped (`pytest.txt`) |

smoke log 폴더는 삭제 권한이 없어 남아 있습니다 (각 38 MB):

```
rm -r ~/ame_mjlab_sgs/logs/rsl_rl/ame_g1/2026-11-22_19-24-49_smoke_amp_task1 ~/ame_mjlab_sgs/logs/rsl_rl/ame_g1/2026-11-22_19-30-53_smoke_amp_task2
```

## 4. 실행 명령 (실행하지 않음)

이 PC, 태스크 1:

```
cd ~/ame_mjlab_sgs && nohup setsid uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-AMP-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name v2_progress_amp_1024 --agent.logger tensorboard > ~/train_v2_progress_amp_1024.log 2>&1 < /dev/null &
```

노트북, 태스크 2:

```
# 1) 코드
cd ~/ame_mjlab_sgs
git remote get-url mine || git remote add mine https://github.com/kjin33862/ame_mjlab.git
git fetch mine amp && git checkout -b amp mine/amp    # 이미 있으면: git checkout amp && git pull mine amp
uv sync

# 2) 모션 데이터 (저장소 밖. 코드는 ~/ref/amp_mjlab_npz/amp 의 npz 17개를 읽음)
git clone https://github.com/ccrpRepo/AMP_mjlab ~/ref/AMP_mjlab && git -C ~/ref/AMP_mjlab checkout 6c7a294
mkdir -p ~/ref/amp_mjlab_npz/amp
uv run python ~/ref/AMP_mjlab/scripts/csv_to_npz.py --input-dir ~/ref/AMP_mjlab/motion_data_csv/amp --output-dir ~/ref/amp_mjlab_npz/amp --input-fps 120 --output-fps 50
ls ~/ref/amp_mjlab_npz/amp/*.npz | wc -l   # 17
#    (변환 대신 이 PC의 ~/ref/amp_mjlab_npz 를 그대로 복사해도 됨. 다른 경로면 AME_AMP_MOTION_DIR=<경로>)

# 3) 학습
nohup setsid uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-AME2Contact-FootholdD30-AMP-Velocity-Rough-G1 --env.scene.num-envs 512 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name d30_amp_ame2contact_512 --agent.logger tensorboard > ~/train_d30_amp_ame2contact_512.log 2>&1 < /dev/null &
```

## 5. 우려

1. **style reward가 약할 수 있음.** 기본값 그대로면 group 1에서 style 대 task의 비율은 원래 단위로 1 : 3입니다 (0.25·r_s 대 0.75·r1). smoke에서 style은 step당 약 0.013 (최대 0.1)입니다. reference의 task reward도 mjlab dt-scale이라 단위는 같지만, 우리 보상 stack의 크기는 다릅니다. 2k it 전후에 `Loss/amp_style_reward`가 오르는지, hopping(동시 착지)이 줄어드는지 확인하길 권합니다 (notes/amp_gait_check의 측정을 다시 쓰면 됨).
2. **D update 횟수와 batch.** reference 규칙(PPO minibatch당 1회)을 따라 우리 minibatch 16 × 5 epoch = 80회, batch 1536입니다 (reference 4096 envs 기준 20회 × 24576). iteration당 본 샘플 수(5 pass)는 같지만, step이 더 많고 batch가 작아서 D가 더 빨리 강해질 수 있습니다. lr도 우리 adaptive(desired_kl 0.02)를 따릅니다 (reference는 0.01).
3. **평지 데이터만 있음.** 계단·돌에서 필요한 동작(높은 무릎, 짧은 보폭)도 D가 "데이터 밖"으로 볼 수 있습니다. AMP 관측에는 지형 정보가 없습니다.
4. **jog clip에는 양발 공중 구간이 있음** (평균 0.41). AMP는 비행 자체가 아니라 데이터에 없는 양발 동시 착지와 짧은 공중 시간을 벌하게 됩니다.
5. 재현한 reference의 특이점: gradient penalty를 정규화 전 입력에서 계산하고, 종료 transition의 next = 직전 frame입니다.
6. **태스크 2.** 400 N은 AME-2 로봇 기준입니다 (우리 G1은 327 N). pelvis_contour_link 대신 pelvis를 씁니다. 무릎·손 접촉이 허용되므로 무릎으로 기는 동작이 나올 수 있습니다 (감점은 step당 −1·0.02 × 접촉 link 수). `termination_penalty` −50은 이제 fell_over와 torso_collision에만 걸립니다.
7. checkpoint의 optimizer에 D param group이 있어서, AMP가 없는 checkpoint의 optimizer 상태는 AMP 태스크로 이어 학습할 수 없습니다 (actor/critic만 로드하면 됨). 반대로 AMP checkpoint의 actor는 기존 평가 스크립트(actor only)로 로드됩니다.
8. 노트북에서의 속도와 메모리는 측정하지 않았습니다 (이 PC 1024 envs: 약 10.4 s/it).

파일: `notes/amp_check/` 아래 `config_hashes.py`, `config_hashes_{old,new,diff}.txt`, `replay_check.py`, `shape_check.py`, `knee_probe.py`, `knee_probe.txt`, `pytest.txt`, 그림 2개.

**push.** 모든 검사가 통과해서 `git push mine amp`를 시도했지만, 이 PC에 GitHub 자격 증명이 없어 실패했습니다 (`could not read Username for 'https://github.com'`). 커밋은 로컬 branch `amp`에 있습니다. 자격 증명을 설정한 뒤 `cd ~/ame_mjlab_sgs && git push mine amp`를 실행하면 됩니다. 노트북의 1) 단계는 push 이후에 가능합니다.
