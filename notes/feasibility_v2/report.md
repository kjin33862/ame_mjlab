notes/feasibility_v2/ — 71cf35f GoalReach report: note the push needs credentials

# Foothold v2 / Progress shortfall 타당성 검토 (분석만, 학습·등록·커밋 없음)

체크포인트 `logs/rsl_rl/ame_g1/2026-11-17_21-45-52_foothold_d30_alphanist_mix02_1024/model_8600.pt`, 태스크 Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1, actor만, noise off, 조건당 16대 × 10 s, seed 42. 측정 `notes/edge_measure/measure_v2.py`(reward_audit.py 복사 + sole scan·contact 로그), 분석 `notes/feasibility_v2/analyze_v2.py`.

## 결론 먼저
- **추천: δ = 0.05 m, 제곱(x²).**
  - 평지·wave L2에서 오탐 0%이고, 계단 L1–5(riser 6.7 cm 이상)와 돌을 δ 0.02와 똑같이 검출합니다.
  - δ 0.02/0.03은 wave·rough에서 오탐이 30–95%입니다. δ 0.08은 오탐이 1%지만 **계단 L1(6.7 cm)을 전혀 못 잡습니다**(up 1%, down 0%).
  - δ 0.05의 남은 오탐은 경사 0.3 m/m 이상의 touchdown입니다(wave L9 34%, rough 40%). 다만 제곱이라 초당 −0.07(wave L9), −0.15(rough)로 작습니다. 제곱은 선형보다 오탐 비용을 1/3–1/4로 줄입니다.
  - 측정하지 않았지만 δ 0.06이면 L1을 잡으면서 오탐을 더 줄일 가능성이 있습니다.
- **피할 수 없는 계단 레벨은 없습니다.**
  - tread는 0.30 m이고 발 capsule 길이는 0.206 m라서, 모든 레벨에서 발 전체를 한 tread에 올릴 수 있습니다(여유 9.4 cm).
  - 지금 정책도 오르막 touchdown의 44–56%는 이미 한 tread 위에 있습니다.
  - 반면 **내리막은 12–18%만 한 tread 위**이고 x>0이 89–94%입니다(뒤꿈치나 앞꿈치가 모서리 밖으로 나감). v2를 넣으면 내리막에서 비용이 가장 큽니다(제곱 −0.27~−0.34/s).
- **(iii)이 이동/걷기를 확실히 낫게 만드는가:**
  - **평지에서는 그렇습니다.** 걷기 − 서 있기 차이가 +1.20(i) → +2.63/s(ii, iii)로 커집니다. progress가 서 있는 쪽에만 −1.44/s를 줍니다.
  - **stones L2–3에서는 아닙니다.** 이동 − 멈춤 차이가 −0.11(i) → +0.48(ii) → **+0.05/s(iii)**입니다. v2(제곱, δ0.05)가 이동 구간에 −0.58/s를 줘서(현재 foothold −0.15보다 4배) progress로 생긴 이득을 거의 다 상쇄합니다. 멈춤 구간에는 touchdown이 없어 v2가 0입니다. 그래서 (iii)에서는 돌 위에서 멈추는 것과 움직이는 것이 거의 같은 값입니다.
  - 주의: stones 멈춤 구간 표본이 31 robot-s뿐이고 run마다 달라서, 이 차이는 ±0.1–0.2 정도의 불확실성이 있다고 봐야 합니다.
- **해가 될 수 있는 곳:**
  1. **계단 16–17 cm에서 progress가 −0.64, −1.21/s**입니다. 이 정책은 17 cm를 거의 오르지 못합니다(속도 0.16 m/s). 못 오르는 지형에서 큰 지속 벌점을 주는 셈이라 curriculum 상단에서 학습 신호가 거칠어질 수 있습니다.
  2. **flat vx0.5에서 progress −0.30/s**입니다. 정책이 0.5 명령에 0.32 m/s만 내서 ratio가 0.63이기 때문입니다. 느린 명령의 추종 부족을 tracking과 이중으로 벌하는 셈입니다.
  3. **v2(δ0.05)는 rough·wave L9의 가파른 경사를 오탐**합니다(x>0 40%, 34%).
  4. **v2는 계단 내리막에서 거의 매 touchdown을 벌합니다**(약 90%). 학습하면 내리막 보폭과 착지 위치가 바뀌어야 하므로, 그 대가로 속도 저하나 멈춤이 생기는지 봐야 합니다.

## A. 사실 (file:line)
- **feet_ground_contact 센서**:
  - 정의: `.venv/lib/python3.13/site-packages/mjlab/tasks/velocity/config/g1/env_cfgs.py:56-67`. primary는 `left/right_ankle_roll_link` subtree, secondary는 `terrain` body, fields `found`, `force`, `reduce="netforce"`, `num_slots=1`, `track_air_time=True`.
  - contact flag는 `found > 0`이고 **힘 임계값이 없습니다**: `mjlab/sensor/contact_sensor.py:462` (air-time 계산)와 `src/ame/mdp/foothold.py:159-165` (`feet_in_contact`). 그래서 stance 중 0.01 s 깜빡임이 first-contact로 잡힙니다(audit 기준 55–66%).
- **계단** (학습 보드 `gap_board_cfg`, `src/ame/terrains.py:363`, 사용처 `src/ame/env_cfg.py:709`):
  - `pyramid_stairs` / `pyramid_stairs_inv`: `step_height_range=(0.05, 0.20)`, `step_width=0.30` (`src/ame/terrains.py:399-404`).
  - 기본값 `platform_width=3.0`, `border_width=1.0`: `mjlab/terrains/config.py:46-68`. 타일 크기 `TILE = (8.0, 8.0)`: `src/ame/terrains.py:32`.
  - 단 수 = int((8 − 2·1 − 3)/(2·0.3)) = **5단**: `mjlab/terrains/primitive_terrains.py:96-104`.
  - riser = 0.05 + 0.15·d (`primitive_terrains.py:91-92`, inv는 `:263-264`), d = row/9 (`mjlab/terrains/terrain_generator.py:263-264`, num_rows 10).
  - **tread 깊이는 모든 레벨 0.30 m**이고 pyr와 inv가 같습니다. riser는 다음과 같습니다.

  | level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | riser (cm) | 5.0 | 6.7 | 8.3 | 10.0 | 11.7 | 13.3 | 15.0 | 16.7 | 18.3 | 20.0 |

- **발바닥 (G1)**: `src/ame/mdp/foothold.py:13-17, 50-57`.
  - 별도 sole box가 없고, 발마다 반경 0.01 m capsule 7개가 `*_ankle_roll_link`의 z = −0.025에 있습니다. sole 평면은 z = −0.035입니다.
  - bounding box는 x ∈ [−0.064, 0.142], y ∈ [−0.037, 0.036]이므로 **길이 0.206 m × 폭 0.073 m**입니다(오른발 폭 0.072).
  - 15점 grid는 x ∈ [−0.044, 0.123](0.167 m) 5점 × y ∈ {−0.018, 0, 0.018}입니다.

## B/C 정의 (구현 그대로)
- **touchdown**: first contact이면서 직전 swing ≥ 0.1 s이고, spawn 후 0.5 s 이상 지난 시점.
- **window**: touchdown step부터 +4 step(0.1 s)까지. 발이 떨어지거나 리셋되면 거기서 자릅니다.
- **off 판정**: `foot_sole_scan` hit z(발 기울기 무시)가 15점 최고값 − δ보다 낮으면 off. ray miss(돌 사이 2 m 구덩이 등)도 off입니다.
- **x**: off 비율을 window 안에서 최댓값으로 잡은 값입니다. 벌점은 touchdown당 −x 또는 −x²(weight −1)이고, 초당 값은 합을 robot-seconds로 나눈 것입니다.
- **progress**: spawn 후 1 s 이상 지났고 |v_cmd| > 0.3일 때만 계산합니다. 그 전 1 s는 계산하지 않습니다(같은 에피소드 안의 1 s 창만 쓰기 위해서). 명령 방향은 body command를 현재 heading으로 돌려서 얻습니다.
- **(iii)**: 현재 foothold term(δ 0.30, stance 매 step)을 빼고, v2 제곱(δ 0.05, touchdown step에 부여)을 더한 값입니다.

## 측정 못 한 것 / 주의
- stones 멈춤/이동 구간은 GPU 비결정성 때문에 run마다 달라집니다. audit에서는 L2 21.6% 멈춤이었고 이번에는 8.2%였습니다. 멈춤 표본이 31 robot-s뿐이라 (c) 비교의 불확실성이 큽니다.
- random_rough는 mjlab에서 difficulty를 쓰지 않아 레벨 구분 없이 한 조건만 측정했습니다.
- 계단 up/down 구분은 발 위치로 했습니다: 타일 중심에서 1.35–3.15 m 구간이고, +x로 걸을 때 바깥쪽인지로 판정합니다. 초당 값은 그 레벨 32대(pyr + inv) 전체 시간으로 나눴습니다.
- 현재 foothold(δ 0.30)는 계단과 wave에서 사실상 0입니다(돌에서만 −0.12~−0.20/s). 그래서 이 두 지형에서는 v2가 새로 생기는 비용입니다.
- v2는 "한 번 학습한 뒤의 정책 행동"이 아니라 **현재 정책 기준의 비용**입니다. 학습하면 행동이 바뀌어 수치도 달라집니다.
- δ 0.06과 다른 window 길이는 시험하지 않았습니다.

## 표 (analyze_v2.py 출력 그대로)
## B. Foothold v2 — 지형별 (vx 1.0, 16대; 계단은 pyr+inv 32대의 해당 구간 touchdown)

per second 값 = weight −1 × Σ(x 또는 x²) / robot-seconds (계단 구간은 그 레벨 32대 전체 robot-seconds로 나눔). 현재 foothold = 학습 term(δ 0.30, stance 매 step)의 초당 값(조건 전체).

### delta = 0.02 m

| 지형 | touchdown | x>0 비율 | x≥0.5 비율 | 선형 /s | 제곱 /s | 현재 foothold /s |
|---|---:|---:|---:|---:|---:|---:|
| flat | 639 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L2 (4.4 cm) | 636 | 0.024 | 0.006 | -0.035 | -0.015 | +0.000 |
| wave L5 (11.1 cm) | 620 | 0.658 | 0.037 | -0.884 | -0.351 | -0.000 |
| wave L9 (20 cm) | 619 | 0.819 | 0.544 | -1.697 | -0.947 | -0.000 |
| random_rough | 645 | 0.950 | 0.592 | -2.189 | -1.341 | -0.000 |
| stairs up L1 (6.7 cm) | 311 | 0.675 | 0.424 | -0.403 | -0.273 | +0.000 |
| stairs up L2 (8.3 cm) | 300 | 0.647 | 0.383 | -0.354 | -0.233 | +0.000 |
| stairs up L3 (10.0 cm) | 294 | 0.592 | 0.323 | -0.306 | -0.198 | -0.000 |
| stairs up L4 (11.7 cm) | 313 | 0.508 | 0.291 | -0.283 | -0.185 | +0.000 |
| stairs up L5 (13.3 cm) | 311 | 0.505 | 0.267 | -0.278 | -0.178 | -0.000 |
| stairs down L1 (6.7 cm) | 247 | 0.903 | 0.591 | -0.418 | -0.271 | +0.000 |
| stairs down L2 (8.3 cm) | 257 | 0.938 | 0.564 | -0.430 | -0.269 | +0.000 |
| stairs down L3 (10.0 cm) | 274 | 0.931 | 0.522 | -0.457 | -0.290 | -0.000 |
| stairs down L4 (11.7 cm) | 297 | 0.892 | 0.508 | -0.476 | -0.307 | +0.000 |
| stairs down L5 (13.3 cm) | 319 | 0.900 | 0.555 | -0.525 | -0.339 | -0.000 |
| stones L1 (gap 5 cm) | 573 | 0.607 | 0.389 | -1.410 | -1.058 | -0.204 |
| stones L2 (gap 10 cm) | 382 | 0.492 | 0.285 | -0.745 | -0.505 | -0.122 |
| stones L3 (gap 15 cm) | 299 | 0.699 | 0.528 | -1.027 | -0.771 | -0.142 |

### delta = 0.03 m

| 지형 | touchdown | x>0 비율 | x≥0.5 비율 | 선형 /s | 제곱 /s | 현재 foothold /s |
|---|---:|---:|---:|---:|---:|---:|
| flat | 639 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L2 (4.4 cm) | 636 | 0.013 | 0.002 | -0.014 | -0.005 | +0.000 |
| wave L5 (11.1 cm) | 620 | 0.295 | 0.027 | -0.236 | -0.090 | -0.000 |
| wave L9 (20 cm) | 619 | 0.693 | 0.110 | -1.061 | -0.450 | -0.000 |
| random_rough | 645 | 0.839 | 0.250 | -1.403 | -0.689 | -0.000 |
| stairs up L1 (6.7 cm) | 311 | 0.675 | 0.424 | -0.403 | -0.273 | +0.000 |
| stairs up L2 (8.3 cm) | 300 | 0.647 | 0.383 | -0.354 | -0.233 | +0.000 |
| stairs up L3 (10.0 cm) | 294 | 0.592 | 0.323 | -0.306 | -0.198 | -0.000 |
| stairs up L4 (11.7 cm) | 313 | 0.508 | 0.291 | -0.283 | -0.185 | +0.000 |
| stairs up L5 (13.3 cm) | 311 | 0.505 | 0.267 | -0.278 | -0.178 | -0.000 |
| stairs down L1 (6.7 cm) | 247 | 0.903 | 0.591 | -0.418 | -0.271 | +0.000 |
| stairs down L2 (8.3 cm) | 257 | 0.938 | 0.564 | -0.430 | -0.269 | +0.000 |
| stairs down L3 (10.0 cm) | 274 | 0.931 | 0.522 | -0.457 | -0.290 | -0.000 |
| stairs down L4 (11.7 cm) | 297 | 0.892 | 0.508 | -0.476 | -0.307 | +0.000 |
| stairs down L5 (13.3 cm) | 319 | 0.900 | 0.555 | -0.525 | -0.339 | -0.000 |
| stones L1 (gap 5 cm) | 573 | 0.607 | 0.389 | -1.410 | -1.058 | -0.204 |
| stones L2 (gap 10 cm) | 382 | 0.492 | 0.285 | -0.745 | -0.505 | -0.122 |
| stones L3 (gap 15 cm) | 299 | 0.699 | 0.528 | -1.026 | -0.769 | -0.142 |

### delta = 0.05 m

| 지형 | touchdown | x>0 비율 | x≥0.5 비율 | 선형 /s | 제곱 /s | 현재 foothold /s |
|---|---:|---:|---:|---:|---:|---:|
| flat | 639 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L2 (4.4 cm) | 636 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L5 (11.1 cm) | 620 | 0.031 | 0.019 | -0.062 | -0.034 | -0.000 |
| wave L9 (20 cm) | 619 | 0.339 | 0.008 | -0.263 | -0.065 | -0.000 |
| random_rough | 645 | 0.397 | 0.043 | -0.434 | -0.153 | -0.000 |
| stairs up L1 (6.7 cm) | 311 | 0.675 | 0.424 | -0.403 | -0.273 | +0.000 |
| stairs up L2 (8.3 cm) | 300 | 0.647 | 0.383 | -0.354 | -0.233 | +0.000 |
| stairs up L3 (10.0 cm) | 294 | 0.592 | 0.323 | -0.306 | -0.198 | -0.000 |
| stairs up L4 (11.7 cm) | 313 | 0.508 | 0.291 | -0.283 | -0.185 | +0.000 |
| stairs up L5 (13.3 cm) | 311 | 0.505 | 0.267 | -0.278 | -0.178 | -0.000 |
| stairs down L1 (6.7 cm) | 247 | 0.903 | 0.591 | -0.418 | -0.271 | +0.000 |
| stairs down L2 (8.3 cm) | 257 | 0.938 | 0.564 | -0.430 | -0.269 | +0.000 |
| stairs down L3 (10.0 cm) | 274 | 0.931 | 0.522 | -0.457 | -0.290 | -0.000 |
| stairs down L4 (11.7 cm) | 297 | 0.892 | 0.508 | -0.476 | -0.307 | +0.000 |
| stairs down L5 (13.3 cm) | 319 | 0.900 | 0.555 | -0.525 | -0.339 | -0.000 |
| stones L1 (gap 5 cm) | 573 | 0.607 | 0.363 | -1.348 | -0.978 | -0.204 |
| stones L2 (gap 10 cm) | 382 | 0.492 | 0.280 | -0.728 | -0.481 | -0.122 |
| stones L3 (gap 15 cm) | 299 | 0.699 | 0.515 | -0.994 | -0.726 | -0.142 |

### delta = 0.08 m

| 지형 | touchdown | x>0 비율 | x≥0.5 비율 | 선형 /s | 제곱 /s | 현재 foothold /s |
|---|---:|---:|---:|---:|---:|---:|
| flat | 639 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L2 (4.4 cm) | 636 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| wave L5 (11.1 cm) | 620 | 0.019 | 0.010 | -0.036 | -0.018 | -0.000 |
| wave L9 (20 cm) | 619 | 0.010 | 0.006 | -0.020 | -0.012 | -0.000 |
| random_rough | 645 | 0.012 | 0.003 | -0.020 | -0.009 | -0.000 |
| stairs up L1 (6.7 cm) | 311 | 0.010 | 0.000 | -0.001 | -0.000 | +0.000 |
| stairs up L2 (8.3 cm) | 300 | 0.647 | 0.383 | -0.354 | -0.233 | +0.000 |
| stairs up L3 (10.0 cm) | 294 | 0.592 | 0.323 | -0.306 | -0.198 | -0.000 |
| stairs up L4 (11.7 cm) | 313 | 0.508 | 0.291 | -0.283 | -0.185 | +0.000 |
| stairs up L5 (13.3 cm) | 311 | 0.505 | 0.267 | -0.278 | -0.178 | -0.000 |
| stairs down L1 (6.7 cm) | 247 | 0.000 | 0.000 | -0.000 | -0.000 | +0.000 |
| stairs down L2 (8.3 cm) | 257 | 0.938 | 0.564 | -0.430 | -0.269 | +0.000 |
| stairs down L3 (10.0 cm) | 274 | 0.931 | 0.522 | -0.457 | -0.290 | -0.000 |
| stairs down L4 (11.7 cm) | 297 | 0.892 | 0.508 | -0.476 | -0.307 | +0.000 |
| stairs down L5 (13.3 cm) | 319 | 0.900 | 0.555 | -0.525 | -0.339 | -0.000 |
| stones L1 (gap 5 cm) | 573 | 0.586 | 0.161 | -0.878 | -0.481 | -0.204 |
| stones L2 (gap 10 cm) | 382 | 0.466 | 0.173 | -0.547 | -0.287 | -0.122 |
| stones L3 (gap 15 cm) | 299 | 0.682 | 0.401 | -0.827 | -0.537 | -0.142 |

### wave / random_rough / flat: x>0 비율 vs touchdown 시 국소 경사 (15점 평면 fit 기울기)

| 경사 구간 (m/m) | touchdown | δ=0.02 | δ=0.03 | δ=0.05 | δ=0.08 |
|---|---:|---:|---:|---:|---:|
| 0.00–0.05 | 900 | 0.013 | 0.011 | 0.001 | 0.000 |
| 0.05–0.1 | 596 | 0.094 | 0.057 | 0.012 | 0.002 |
| 0.10–0.2 | 662 | 0.737 | 0.301 | 0.048 | 0.000 |
| 0.20–0.3 | 505 | 0.976 | 0.869 | 0.160 | 0.004 |
| 0.30–0.5 | 440 | 0.995 | 0.964 | 0.734 | 0.014 |
| 0.50–∞ | 55 | 1.000 | 0.982 | 0.727 | 0.291 |

15점 길이(x 방향) 0.167 m: 경사 s에서 앞뒤 끝 높이차 ≈ 0.167·s → δ보다 크면 평평한 경사에서도 off. (δ=0.02 → s>0.12, 0.03 → 0.18, 0.05 → 0.30, 0.08 → 0.48)

### 계단: 한 tread에 발 전체가 놓인 touchdown 비율 (touchdown step의 15점 높이 span < 1 cm)

| 레벨 | riser (cm) | up: touchdown / 한 tread 비율 | down: touchdown / 한 tread 비율 | up: x>0 (δ0.05) | down: x>0 (δ0.05) |
|---|---:|---|---|---:|---:|
| L1 | 6.7 | 311 / 0.44 | 247 / 0.17 | 0.68 | 0.90 |
| L2 | 8.3 | 300 / 0.46 | 257 / 0.12 | 0.65 | 0.94 |
| L3 | 10.0 | 294 / 0.46 | 274 / 0.15 | 0.59 | 0.93 |
| L4 | 11.7 | 313 / 0.56 | 297 / 0.18 | 0.51 | 0.89 |
| L5 | 13.3 | 311 / 0.55 | 319 / 0.16 | 0.50 | 0.90 |

계단 tread 0.30 m > 발 capsule 길이 0.206 m(15점 0.167 m): 기하적으로 모든 레벨에서 발 전체를 한 tread에 올릴 수 있음(여유 0.094 m).

## C. Progress shortfall (weight −2, 초당 평균)

| 조건 | progress /s | 그 중 penalty>0 인 step 비율 | 평균 ratio (계산된 step) |
|---|---:|---:|---:|
| flat_v0.5 | -0.298 | 0.90 | 0.63* |
| flat_v1.0 | -0.015 | 0.07 | 0.79* |
| flat_v1.2 | -0.016 | 0.06 | 0.79* |
| flat_stand_policy | -1.440 | 0.90 | 0.00* |
| stones L2-3 moving | -0.361 | 0.42 | 0.56* |
| stones L2-3 stopped | -0.953 | 0.62 | 0.03* |
| stairs up 10 cm | -0.041 | 0.29 | 0.78* |
| stairs up 11 cm | -0.093 | 0.48 | 0.75* |
| stairs up 12 cm | -0.100 | 0.47 | 0.74* |
| stairs up 13 cm | -0.130 | 0.49 | 0.73* |
| stairs up 14 cm | -0.238 | 0.60 | 0.67* |
| stairs up 15 cm | -0.331 | 0.66 | 0.62* |
| stairs up 16 cm | -0.642 | 0.73 | 0.44* |
| stairs up 17 cm | -1.205 | 0.89 | 0.13* |

*penalty=max(0,0.8−ratio)라서 penalty>0 구간에서만 ratio가 복원됨; 이 열은 0.8−penalty 평균(=min(ratio,0.8) 평균).

## Q1 재계산: (i) 현재 reward, (ii) + progress(−2), (iii) + progress, foothold → v2(δ=0.05, 제곱, −1/touchdown)

| 구간 | robot-s | (i) 현재 total /s | (ii) +progress | (iii) +progress, v2 | 현재 foothold /s | v2 /s |
|---|---:|---:|---:|---:|---:|---:|
| flat_v1.0 | 160 | +7.144 | +7.129 | +7.129 | +0.000 | +0.000 |
| flat_stand_policy | 160 | +5.944 | +4.504 | +4.504 | +0.000 | +0.000 |
| stones L2-3 moving | 289 | +5.713 | +5.352 | +4.918 | -0.146 | -0.580 |
| stones L2-3 stopped | 31 | +5.822 | +4.868 | +4.868 | +0.000 | +0.000 |
| flat_v0.5 | 160 | +7.170 | +6.872 | +6.872 | +0.000 | +0.000 |
| flat_v1.2 | 160 | +7.060 | +7.045 | +7.045 | +0.000 | +0.000 |

- flat_v1.0 − flat_stand_policy: (i) +1.200, (ii) +2.625, (iii) +2.625 /s
- stones L2-3 moving − stones L2-3 stopped: (i) -0.109, (ii) +0.484, (iii) +0.050 /s
- stones_L2: stopped 8.2% of robot-time
- stones_L3: stopped 11.3% of robot-time

### progress 전체 조건 (progress_other_terrains.txt)
```
progress shortfall, weight -2, per second (16 robots, 10 s)
flat_v0.5          -0.298  share penalised 0.90  mean speed 0.32  terminations 0
flat_v1.0          -0.015  share penalised 0.07  mean speed 0.85  terminations 0
flat_v1.2          -0.016  share penalised 0.06  mean speed 1.03  terminations 0
flat_stand_policy  -1.440  share penalised 0.90  mean speed 0.01  terminations 0
wave_L2            -0.031  share penalised 0.23  mean speed 0.82  terminations 0
wave_L5            -0.052  share penalised 0.22  mean speed 0.81  terminations 2
wave_L9            -0.138  share penalised 0.54  mean speed 0.75  terminations 1
random_rough       -0.112  share penalised 0.38  mean speed 0.78  terminations 0
pyr_L1             -0.031  share penalised 0.19  mean speed 0.85  terminations 0
pyr_L2             -0.036  share penalised 0.24  mean speed 0.84  terminations 0
pyr_L3             -0.051  share penalised 0.33  mean speed 0.83  terminations 0
pyr_L4             -0.097  share penalised 0.49  mean speed 0.78  terminations 0
pyr_L5             -0.161  share penalised 0.54  mean speed 0.75  terminations 1
inv_L1             -0.033  share penalised 0.22  mean speed 0.84  terminations 0
inv_L2             -0.026  share penalised 0.21  mean speed 0.85  terminations 0
inv_L3             -0.042  share penalised 0.34  mean speed 0.82  terminations 0
inv_L4             -0.102  share penalised 0.46  mean speed 0.79  terminations 0
inv_L5             -0.184  share penalised 0.54  mean speed 0.75  terminations 0
stones_L1          -0.025  share penalised 0.13  mean speed 0.93  terminations 0
stones_L2          -0.347  share penalised 0.37  mean speed 0.66  terminations 18
stones_L3          -0.491  share penalised 0.51  mean speed 0.51  terminations 35
stairs_10          -0.041  share penalised 0.29  mean speed 0.84  terminations 0
stairs_11          -0.093  share penalised 0.48  mean speed 0.78  terminations 0
stairs_12          -0.100  share penalised 0.47  mean speed 0.79  terminations 0
stairs_13          -0.130  share penalised 0.49  mean speed 0.77  terminations 0
stairs_14          -0.238  share penalised 0.60  mean speed 0.71  terminations 0
stairs_15          -0.331  share penalised 0.66  mean speed 0.66  terminations 0
stairs_16          -0.642  share penalised 0.73  mean speed 0.52  terminations 1
stairs_17          -1.205  share penalised 0.89  mean speed 0.16  terminations 0
```

## 그림
- `fig_b_foothold_v2.png`: 지형별 x>0 비율(δ 4개), wave·rough·flat의 경사별 오탐.
- `fig_c_totals.png`: (i)/(ii)/(iii) 초당 total. 평지 걷기 vs 서 있기, stones 이동 vs 멈춤.

## 파일
`notes/edge_measure/measure_v2.py`, `notes/edge_measure/raw_v2.npz`(원자료, 68 MB), `notes/feasibility_v2/analyze_v2.py`, `tables_v2.md`, `progress_other_terrains.txt`, 그림 2개. 커밋 안 함.
