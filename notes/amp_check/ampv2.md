# AMPv2 (D30 + 레퍼런스 판별기 스케줄) — 구현·스모크·런치

## 결론
- 새 태스크 `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-AMPv2-Velocity-Rough-G1` 등록. env/play config는 D30과 해시 동일, 기존 35개 태스크 해시 모두 불변.
- 판별기: 별도 Adam, **고정 LR 1e-3**, 반복당 **20회** 업데이트, 미니배치 **6144** (1024 env × 24 / 4). 스모크에서 실제 로그로 확인 (`Loss/amp_disc_updates`=20, `amp_disc_mini_batch_size`=6144, `amp_disc_lr`=0.001, 30회 내내 고정).
- 스모크 30 it: 에러·NaN 없음. 그러나 **D(expert)≈+0.84 / D(policy)≈−0.84로 빠르게 포화하는 양상은 이전 런과 같음** (아래 표). 스타일 보상은 0.017~0.022 (이전 0.013). 판별기 스케줄만으로는 포화가 풀리지 않을 수 있음 — 효과는 lerp 0.5로 커진 스타일 비중에 달려 있음.
- `d30_ampv2_1024` 런치 완료 (2026-11-23 05:01 PST), 커밋 `d967824`, `mine/amp`에 push 완료.
- 속도 **≈10.0 s/it**, ETA **≈27.8 h** (로그 ETA "1 day, 3:50"; 2026-11-24 09시경 PST 종료 예상).

## 주의: 레퍼런스의 판별기 LR은 사실 "고정"이 아님
레퍼런스는 판별기를 PPO와 **같은 Adam**에 넣고 (`rsl_rl/algorithms/amp_ppo.py:127-132`), 설정이 `schedule="adaptive"`, `desired_kl=0.01` (`src/tasks/amp_loco/config/g1/rl_cfg.py:60,63`)이므로, 판별기 LR도 KL 적응 LR을 따라 움직임 (`amp_ppo.py:327-361`, 모든 param group에 적용). `schedule="fixed"`는 AMPPPO 생성자 기본값일 뿐 (`amp_ppo.py:57`)이고 config가 덮어씀.
요청대로 **고정 LR**로 구현했고, 값은 레퍼런스 초기 LR 1e-3 (`rl_cfg.py:59`, `amp_ppo.py:54`). 레퍼런스를 문자 그대로 따르면 PPO의 desired_kl을 0.01로 바꿔야 해서 (우리는 0.02) D30 베이스가 바뀌므로 채택하지 않음.

## 레퍼런스에서 가져온 값 (AMP_mjlab @ 6c7a294)
| 항목 | 값 | 출처 |
|---|---|---|
| 반복당 판별기 업데이트 | epochs × minibatches = 5 × 4 = 20 | `rl_cfg.py:57-58`, `amp_ppo.py:252-262` (PPO 미니배치마다 1스텝, zip) |
| 판별기 미니배치 | num_envs × num_steps_per_env // num_mini_batches = 1024×24/4 = 6144 | `amp_ppo.py:253-258`, `rl_cfg.py:70` (24) |
| 옵티마이저 | Adam (기본 betas/eps) | `amp_ppo.py:132` |
| LR | 1e-3 (여기선 고정, 위 주의 참조) | `rl_cfg.py:59` |
| weight decay | trunk 10e-4 = 1e-3, head 10e-2 = 1e-1 | `amp_ppo.py:129-130` |
| grad clip | 없음 (정책만 clip) | `amp_ppo.py:467` |
| 손실 | 0.5·(MSE(D_e,1)+MSE(D_p,−1)), 계수 1.0 | `amp_ppo.py:114,446-450` |
| gradient penalty | λ=10, 정규화 전 expert 쌍 | `amp_ppo.py:449`, `rsl_rl/modules/discriminator.py:84-111` |
| 정규화기 업데이트 | 스텝마다 policy·expert state | `amp_ppo.py:496-498` |
| replay buffer | 100000 (config 미지정 → 기본값) | `amp_ppo.py:45,119` |
| 판별기 구조 | 1024-512-256 | `rl_cfg.py:77` |
| 스타일 보상 계수 | 0.1 | `rl_cfg.py:73` |
| task_reward_lerp | 레퍼런스 0.75 → **0.5** (유일한 의도적 차이) | `rl_cfg.py:76`, `discriminator.py:157` |

## 바뀐 코드 (커밋 d967824)
- `src/ame/amp.py:300-340` — 기본 AMP 클래스를 훅으로 분리 (`_attach_discriminator_optimizer`, `_disc_loss`, `_rollout_stats`). 기존 AMP 태스크의 동작은 동일 (같은 연산, 같은 순서).
- `src/ame/amp.py:460-575` — `AmpV2DoubleCriticPPO`: 판별기 전용 Adam (`:507`), PPO 업데이트 후 20스텝 루프 (`:521`, `:547`), 판별기 옵티마이저 state 저장/로드. PPO 옵티마이저에는 정책만 남음 (AMP 없는 D30과 동일).
- `src/ame/rl_cfg.py:257-275` — `AmpV2DoubleCriticPpoCfg` (lerp 0.5, disc lr 1e-3, 5 epochs, 4 minibatches), `ampv2_double_critic_ppo_runner_cfg()` = D30의 `double_critic_ppo_runner_cfg()` + AMPv2. 기존 `AmpDoubleCriticPpoCfg`는 건드리지 않음 (기존 AMP 태스크 해시 유지).
- `src/ame/__init__.py:508-537` — 태스크 등록 (D30과 같은 `ame_gap_env_cfg(...)` 호출).
- `tests/test_amp.py:183,200` — 테스트 2개 추가 (태스크 = D30 + AMPv2, 판별기 스케줄·고정 LR·weight decay). pytest 165 passed, 1 skipped.
- 해시: `notes/amp_check/ampv2_hashes_{old,new}.txt` (diff = 새 태스크 1줄 추가뿐).
- 그 외 그대로: 17개 클립 (`~/ref/amp_mjlab_npz/amp`), AMP 관측 195차원, 스타일 보상은 group 1, foothold는 group 2, advantage_weights (1.0, 0.25).

## 스모크 (1024 env, 30 it, seed 42, 처음부터)
에러·NaN 없음 (모든 스칼라 유한). 9.99~10.0 s/it.

| it | D(expert) | D(policy) | style reward | D(policy) 롤아웃 |
|---|---|---|---|---|
| 0 | 0.450 | −0.550 | 0.0660 | −0.165 |
| 10 | 0.843 | −0.846 | 0.0172 | −0.827 |
| 20 | 0.842 | −0.844 | 0.0221 | −0.763 |
| 29 | 0.824 | −0.821 | 0.0172 | −0.826 |

판별기 업데이트/미니배치/LR: 20 / 6144 / 0.001 (전 구간). PPO LR은 별도로 적응 (1e-5~2.6e-4).
스모크 로그 디렉터리는 삭제함.

## 런치
```
cd ~/ame_mjlab_sgs && nohup setsid uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-AMPv2-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name d30_ampv2_1024 --agent.logger tensorboard > ~/train_d30_ampv2_1024.log 2>&1 < /dev/null &
```
- 로그 디렉터리: `logs/rsl_rl/ame_g1/2026-11-23_05-01-53_d30_ampv2_1024`
- 3분 시점 18 it, 평균 10.0 s/it → 10000 it ≈ 27.8 h.
- 이전 런 `v2_progress_amp_1024`는 중지됨 (로그·체크포인트 보존: `logs/rsl_rl/ame_g1/2026-11-22_21-35-52_v2_progress_amp_1024`).

## 볼 것
`Loss/amp_style_reward`가 0.02 위로 오르는지, `Loss/amp_d_policy`가 −0.8에서 0 쪽으로 움직이는지, 그리고 2k it 즈음 D30 대비 동시 착지/공중시간 (notes/amp_gait_check 방식).
