## 등가성: 새 태스크 training reward manager vs feasibility 표 (초당)

| 조건 | foothold_v2 새 | 표 | 차이 | progress 새 | 표 | 차이 | 종료 수 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat vx1.0 | -0.000 | +0.000 | 둘 다 ≈0 | -0.015 | -0.015 | 둘 다 ≈0 | 0 |
| flat 서 있기 (obs cmd 0, vx1.0 기준) | -0.000 | +0.000 | 둘 다 ≈0 | -1.439 | -1.440 | +0% | 0 |
| stones L2 | -0.297 | -0.481 | +38% | -0.543 | -0.347 | -57% | 9 |
| stones L3 | -0.736 | -0.726 | -1% | -0.449 | -0.491 | +9% | 34 |
| stairs down L3 (pyr+inv, 내리막 구간 touchdown) | -0.323 | -0.290 | -11% | – | – | – | 0 |
| stairs L3 pyr (progress) | – | – | – | -0.047 | -0.051 | +9% | 0 |
| stairs L3 inv (progress) | – | – | – | -0.051 | -0.042 | -21% | 0 |

foothold_v2 '새' = −Σx²/(spawn 후 0.5 s 이후·비종료 robot-seconds), feasibility와 같은 분모. reward manager의 전 step 평균(_step_reward 평균)은 아래.
- flat vx1.0: _step_reward[foothold_v2] 전 step 평균 +0.000/s
- flat 서 있기 (obs cmd 0, vx1.0 기준): _step_reward[foothold_v2] 전 step 평균 +0.000/s
- stones L2: _step_reward[foothold_v2] 전 step 평균 -0.274/s
- stones L3: _step_reward[foothold_v2] 전 step 평균 -0.626/s
- stairs down L3 (pyr+inv, 내리막 구간 touchdown): _step_reward[foothold_v2] 전 step 평균 -0.498/s

## Double critic 분리 (96대 × 500 step)

- group 2 == foothold_v2 × dt: max |차이| 0.00e+00
- group 1 + group 2 == env reward: max |차이| 1.19e-07
- group 1 == env reward − foothold_v2 × dt (progress_shortfall 포함): max |차이| 0.00e+00
- group 2 ≠ 0 인 (step, env): 831 / 48000; 그중 window 종료 step이 아닌 것: 0
- window 종료 step 수: 2545 (x=0으로 0 지급 1714)
- 종료 사유: full 5 sample 2379, lift-off cut 163, 종료/리셋 cut 5
- progress_shortfall ≠ 0 인 (step, env): 20737; 그중 group 2 ≠ 0: 394 (group 2 값은 foothold_v2 × dt와 일치하므로 progress는 group 2에 없음)
- NaN/Inf: rew 0, groups 0
