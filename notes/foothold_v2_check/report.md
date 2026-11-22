notes/foothold_v2_check/ — 코드 커밋 `d782667` (branch `foothold-v2`, push 안 함). 학습은 시작하지 않음.

# FootholdV2-Progress 태스크 구현·검증

## 결론
- `Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1`을 등록했습니다. FootholdD30과 같은 flag에서 `foothold`(δ 0.30)를 빼고 `foothold_v2`(weight −1, group 2)와 `progress_shortfall`(weight −2, group 1)을 넣은 것이고, 그 밖의 env·agent config는 D30과 완전히 같습니다.
- **기존 32개 태스크의 config hash는 모두 그대로**입니다. 새 태스크 한 줄만 추가됐습니다(`config_hashes_diff.txt`). 테스트는 153 passed, 1 skipped입니다(신규 15개).
- **progress_shortfall**은 같은 run에서 feasibility 공식과 ±2.5% 이내로 일치합니다.
- **foothold_v2**는 평지와 계단에서 offline 공식과 같습니다(inv_L3 0.0%, pyr_L3 −2.5%). stones에서는 online이 18–26% 작습니다. reward 시점의 발 pose가 mjlab 구조상 physics substep 하나(5 ms) 이르기 때문입니다(아래 우려 1).
- **double critic**: group 2는 foothold_v2 × dt와 정확히 같고(차이 0), window가 닫히는 step에서만 0이 아닙니다. progress는 group 1에만 들어갑니다.
- **smoke**(1024 envs, 30 it)는 오류나 NaN 없이 끝났고 약 9.8 s/it입니다. **10000 it는 약 27 시간**입니다. 처음부터 학습하는 첫 30 it에서는 에피소드가 4–10 step(0.1–0.2 s)이라, 두 항이 로그에는 찍히지만 ≈0입니다. foothold_v2는 리셋 후 0.5 s, progress는 1 s가 지나야 계산되기 때문입니다. D30 체크포인트에서 이어 학습한 5 it에서는 foothold_v2 −0.039, progress −0.018로 음수이고 0이 아닌 값을 확인했습니다.

## 구현 (file:line)
| 내용 | 위치 |
|---|---|
| off 비율(15점, top − 0.05 미만이면 off, miss = off, 전부 miss = 1) | `src/ame/mdp/foothold_v2.py:88` `sole_off_fraction` |
| touchdown window 상태기계(touchdown step + 4, lift-off·종료 시 cut, x² 지급) | `src/ame/mdp/foothold_v2.py:102` `FootholdWindows`, `:114` `step` |
| foothold_v2 term (touchdown = first contact, air ≥ 0.1 s, 리셋 후 ≥ 0.5 s, 지급 x²/dt → 매니저 ×dt 후 touchdown당 −x²) | `src/ame/mdp/foothold_v2.py:149`, `:170`, cfg `:208` |
| progress 버퍼(1 s, 명령 속도를 매 step heading으로 world 변환해 적분, ratio = d·C/\|C\|² ≥ 0, 평균 명령 속도 > 0.3일 때만, 리셋 시 비움) | `src/ame/mdp/foothold_v2.py:225` `ProgressBuffer`, `:235` |
| progress_shortfall term (raw penalty 반환 → `_step_reward` = −2·penalty = 초당 값) | `src/ame/mdp/foothold_v2.py:263`, `:270`, cfg `:296` |
| foothold 교체 + progress 추가 | `src/ame/env_cfg.py:592` `apply_foothold_v2_progress`, `:605` `ame_gap_foothold_v2_env_cfg` |
| double critic group-2 = `("foothold_v2",)`, advantage_weights (1.0, 0.25) | `src/ame/rl_cfg.py:211`, `:216` |
| 태스크 등록 (train = `FOOTHOLD_D30_TRAIN_FLAGS`, play = `FOOTHOLD_D30_GAP_FLAGS`) | `src/ame/__init__.py:458-467` |
| 테스트 | `tests/test_foothold_v2.py` (off 비율, window 4경우, reset, progress 5경우, D30 대비 config, group, 기존 hash) |

**타이밍 (측정):** reward 시점의 sole-scan hit은 이번 step 값입니다. `env.step` 뒤에 읽은 값과 같습니다(`hit_timing_probe.py`). 그래서 window는 touchdown call부터 5 call을 sample하고 마지막 call에 지급합니다. lift-off step은 sample하지 않습니다. 처음에는 기존 `foothold.py` docstring의 "한 step 늦음"을 따라 한 칸 밀어 구현했습니다. 하지만 같은 run의 offline과 계단에서 12–14%가 어긋나서, 측정으로 확인한 뒤 고쳤습니다. 기존 `foothold.py` docstring의 그 문장은 이 mjlab 버전에서 맞지 않지만, 기존 태스크를 바꾸지 않기 위해 그대로 두었습니다.

## 검사 결과

### 등가성 (model_8600, actor only, noise off, 16대 × 10 s, seed 42, 새 태스크의 training reward manager)
| 조건 | foothold_v2 /s 새 | feasibility 표 | 차이 | progress /s 새 | 표 | 차이 |
|---|---:|---:|---:|---:|---:|---:|
| 평지 vx1.0 | −0.000 | 0.000 | ≈0 | −0.015 | −0.015 | ≈0 |
| 평지 서 있기 (obs cmd 0, vx1.0 기준) | −0.000 | 0.000 | ≈0 | −1.439 | −1.440 | 0% |
| stones L2 | −0.297 | −0.481 | +38% | −0.543 | −0.347 | −57% |
| stones L3 | −0.736 | −0.726 | −1% | −0.449 | −0.491 | +9% |
| stairs down L3 (pyr+inv, 내리막 touchdown) | −0.323 | −0.290 | −11% | – | – | – |
| stairs L3 pyr / inv (progress) | – | – | – | −0.047 / −0.051 | −0.051 / −0.042 | +9% / −21% |

**30%를 넘는 stones L2의 이유** (`same_run.md`):
1. **run마다 행동이 다릅니다.** 같은 seed에서도 GPU 비결정성 때문에 stones L2 종료 수가 run마다 18 / 11 / 5 / 9로 달랐고, 평균 속도도 0.66(feasibility run)과 0.51(이번 run)로 차이가 납니다. 같은 run에 feasibility 공식을 적용하면 progress는 −0.554로 online −0.543과 +1.9% 차이입니다. 그러니 progress 쪽 차이는 전부 행동 차이입니다. foothold_v2 offline 값도 −0.401로 표(−0.481)보다 17% 작습니다.
2. **reward 시점의 5 ms 차이.** 같은 run에서 foothold_v2 online/offline 차이는 stones L2 +26%, L3 +18%이고, 계단 L3는 0%와 −2.5%, 평지는 0입니다. touchdown 수는 같고(332/335), lift-off cut 수도 비슷합니다. 차이는 stones touchdown의 약 12–17%에서 **touchdown step**의 off 비율이 한 열(0.2)만큼 다른 데서 생깁니다. mjlab은 reward를 `sim.forward()` 전에 계산하므로 body pose가 physics substep 하나(5 ms) 이릅니다(`ManagerBasedRlEnv.step`의 주석). 빠르게 돌 모서리에 착지하는 순간에는 이 차이로 발이 1–2 cm 다른 곳에 있습니다. 모든 mjlab reward가 같은 시점의 상태를 보므로 로직 오류는 아닙니다.

### Double critic (96대 × 500 step, `equiv_tables.md`)
| 항목 | 결과 |
|---|---|
| runner `reward_group_terms` / `advantage_weights` | `('foothold_v2',)` / `(1.0, 0.25)` |
| `RewardGroups`의 group-2 열 | `['foothold_v2']`만 |
| group 2 − foothold_v2×dt | max 0 |
| group 1 − (env reward − foothold_v2×dt) | max 0, 즉 progress_shortfall은 group 1에 있음 |
| group 2 ≠ 0인 (step, env) 중 window 종료 step이 아닌 것 | 0 / 831 |
| window 종료 2545회 | full 2379 / lift-off cut 163 / 종료 cut 5 (x=0이라 0 지급 1714) |
| NaN/Inf | 0 |

### Smoke (실제 config, 1024 envs, seed 42)
| 항목 | 30 it, 처음부터 | 5 it, D30 model_8600에서 이어서 (추가 확인) |
|---|---|---|
| 종료 코드 / 오류 / NaN | 0 / 없음 / 0줄 | 0 / 없음 / 0줄 |
| Episode_Reward/foothold_v2 | 로그됨, −0.0001 ~ 0.0000 | 로그됨, 0 → −0.0053 → −0.0127 → −0.0291 → −0.0393 |
| Episode_Reward/progress_shortfall | 로그됨, 0.0000 | 로그됨, 0 → 0 → −0.0047 → −0.0106 → −0.0179 |
| 평균 에피소드 길이 (step) | 10 → 4 | 11 → 94 |
| iteration 시간 | 첫 it 11.9 s, 이후 평균 9.82 s | 9.85 s |
| 10000 it 예상 | 약 27.3 시간 | |

**smoke log dir 두 개가 남아 있습니다.** 삭제 명령이 권한 설정에서 거부되어 지우지 못했습니다(아래 명령 참고).

## 우려
1. **stones의 touchdown step 5 ms 시차** (위 2번). 학습 신호로서는 일관되지만, offline 표보다 stones에서 약 20% 약하게 벌합니다.
2. **처음부터 학습하는 초기에는 두 항이 ≈0입니다.** 에피소드가 0.5–1 s를 넘어야 작동하므로, 초반 신호는 기존 항들이 주도합니다. 설계대로입니다.
3. feasibility에서 지적한 위험이 그대로 남아 있습니다.
   - 계단 16–17 cm에서 progress −0.6~−1.2/s
   - vx 0.5에서 progress −0.3/s
   - 계단 내리막에서 foothold_v2가 touchdown의 약 90%에 부과
   - rough·wave L9 경사 오탐
4. **stones 비교는 표본이 작습니다**(16대, 멈춤 구간 수십 robot-s). 결론을 바꿀 수준은 아니지만 ±20–50%의 run 간 변동이 있습니다.

## 실행할 명령 (실행하지 않음)
```bash
cd ~/ame_mjlab_sgs && git checkout foothold-v2 && nvidia-smi && free -h
tmux new-session -d -s foothold_v2 "cd ~/ame_mjlab_sgs && uv run train Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name foothold_v2_progress_1024 --agent.logger tensorboard 2>&1 | tee ~/train_foothold_v2_progress_1024.log"
```
smoke log 정리 (권한 거부로 남아 있음):
```bash
rm -rf ~/ame_mjlab_sgs/logs/rsl_rl/ame_g1/2026-11-21_18-47-08_smoke_foothold_v2 ~/ame_mjlab_sgs/logs/rsl_rl/ame_g1/2026-11-21_18-52-56_smoke_foothold_v2_resume
```

## 파일
`equiv_check.py`(측정), `equiv_analyze.py`(표·double critic), `same_run.py`(같은 run offline vs online), `hit_timing_probe.py`(타이밍), `equiv_tables.md`, `same_run.md`, `config_hashes_*.txt`, `pytest.txt`. `equiv.npz`(13 MB)는 커밋하지 않았습니다.
