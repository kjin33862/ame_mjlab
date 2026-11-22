## 같은 run에서 offline 공식(feasibility) vs online term

| 조건 | v2 online /s | v2 offline /s | 차이 | touchdown 수 online / offline | progress online /s | progress offline /s | 차이 |
|---|---:|---:|---:|---|---:|---:|---:|
| flat_v1.0 | -0.000 | -0.000 | ≈0 | 634 / 636 | -0.015 | -0.014 | ≈0 |
| flat_stand_policy | -0.000 | -0.000 | ≈0 | 0 / 0 | -1.439 | -1.440 | +0.1% |
| stones_L2 | -0.297 | -0.401 | +26.1% | 332 / 335 | -0.543 | -0.554 | +1.9% |
| stones_L3 | -0.736 | -0.901 | +18.3% | 322 / 324 | -0.449 | -0.455 | +1.4% |
| pyr_L3 | -0.545 | -0.532 | -2.5% | 629 / 636 | -0.047 | -0.046 | -0.5% |
| inv_L3 | -0.501 | -0.501 | -0.0% | 630 / 635 | -0.051 | -0.051 | -0.4% |

전부 miss인 window(offline analyze_v2는 건너뜀, 여기서는 online과 같이 x=1): 26개
