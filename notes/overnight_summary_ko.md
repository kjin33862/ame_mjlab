# 밤사이 작업 요약 (2026-10-18)

자세한 기록은 `notes/eval_port_log.md`의 "2026-10-18 — Overnight" 부분에 있습니다.

## 1. 바꾼 것

- **upstream/main 머지**: Shuang의 gap task(`49a85ca`)를 `eval` 브랜치에 머지했습니다 (`d2fa505`, 충돌 없음).
  `uv run pytest` 통과, `uv run list-envs`에 `Ame-Mjlab-Gap-Velocity-Rough-G1`이 나옵니다.
- **성공 기준 변경** (`scripts/evaluate.py`): 이제 **명령 방향(앞쪽)으로 4 m 이상** 가야 성공입니다.
  예전처럼 옆으로 4 m 가는 것은 성공이 아닙니다.
  - 새 열 `forward_at_end_m` (끝났을 때 앞쪽 이동 거리) 추가, `max_xy_disp_m` (아무 방향 최대 거리)는 그대로 둠.
  - 옛 기준이 필요하면 `--success_rule any`.
- **새 평가 지형 `--terrain stone_lattice`**: Shuang의 gap board에 있는 돌 격자와 같은 지형입니다.
  돌 0.40 m, 틈 = 0.45 × 난이도(d) m. 예전 `stepping_stones`는 그대로입니다.
  - 주의: d = 0.25, 1.0에서는 타일 정중앙이 틈(구멍) 위입니다. 그래서 **정중앙에서 가장 가까운 돌 위에서**
    시작하도록 했습니다 (평가 코드만 수정, 학습 코드는 안 건드림).
- `view_eval.py`도 새 지형을 지원합니다. 테스트 추가 (77 passed, 1 skipped).
- 새 스크립트: `scripts/run_policy_matrix.sh` (체크포인트 하나를 전체 평가, 다시 실행하면 끝난 run은 건너뜀),
  `scripts/aggregate_matrix.py` (CSV + 그래프).

## 2. 결과 (로봇 192대 = 8×8 타일 × seed 3개, 노이즈 없음)

성공률, **baseline / shuang_gap2048**:

| 지형 | d=0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| flat | 1.00 / 1.00 | - | - | - | - |
| pyramid_stairs (내려가기) | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 0.96 / 1.00 | 0.87 / 0.94 |
| pyramid_stairs_inv (올라가기) | 1.00 / 0.04 | 1.00 / 0.00 | 0.92 / 0.00 | 0.27 / 0.00 | 0.01 / 0.00 |
| random_rough | 0.95 / 0.96 | 0.96 / 0.98 | 0.93 / 0.96 | 0.94 / 0.97 | 0.96 / 0.98 |
| wave | 1.00 / 0.38 | 1.00 / 1.00 | 0.99 / 1.00 | 0.78 / 0.81 | 0.10 / 0.11 |
| stone_lattice | 1.00 / 1.00 | 0.00 / 0.95 | 0.00 / 0.84 | 0.00 / 0.69 | 0.00 / 0.25 |
| stepping_stones (옛 지형) | 0.00 / 0.67 | 0.01 / 0.98 | 0.00 / 0.97 | 0.00 / 0.54 | 0.00 / 0.16 |

노이즈 실험 (stone_lattice d = 0.5, 한 번에 하나만 바꿈):

| 조건 | baseline | shuang_gap2048 (95% 구간) |
|---|---|---|
| 노이즈 없음 | 0.00 | 0.85 (0.79-0.89) |
| 학습 때 노이즈 | 0.00 | 0.86 (0.80-0.90) |
| 높이 노이즈 0.06 / 0.10 m | 0.00 / 0.00 | 0.81 / 0.87 |
| 높이 offset ±0.10 / ±0.15 m | 0.00 / 0.00 | 0.86 / 0.79 |
| 지도 밀림 1 / 2 / 3칸 | 0.00 / 0.00 / 0.00 | 0.88 / 0.84 / **0.66** |
| 지도 지연 5 / 10 step | 0.00 / 0.00 | 0.78 / **0.70** |

걸음새 (한 발만 땅에 있는 비율): baseline 0.05-0.16 (두 발로 깡충 뜀), Shuang 0.68-0.80 (걷기).

## 3. 그래프가 보여주는 것 (`logs/eval_results/matrix_plots/`)

- `success_vs_difficulty.png`: 돌 지형에서 차이가 가장 큽니다. baseline은 틈이 조금이라도 있으면(d ≥ 0.25) 0%.
  Shuang은 틈이 커질수록 서서히 떨어집니다 (0.95 → 0.25). 대신 **계단 올라가기에서는 Shuang이 0%**, baseline은
  d = 0.5까지 잘 올라갑니다.
- `single_support.png`: 모든 지형에서 baseline은 깡충 뛰고(한 발 비율 낮음), Shuang은 걷습니다.
- `noise_sweep.png`: Shuang은 학습 노이즈 정도는 문제없고, 지도가 3칸(15 cm) 밀리거나 10 step(0.2 s) 늦으면
  성공률이 0.66-0.70으로 떨어집니다. baseline은 노이즈와 상관없이 0%.

동영상 (4 s, stone_lattice d = 0.5):
`logs/eval_videos/baseline/stone_lattice_d0.5_off/rl-video-step-0.mp4` (첫 돌 위에서 제자리 점프),
`logs/eval_videos/shuang_gap2048/stone_lattice_d0.5_off/rl-video-step-0.mp4` (돌을 밟고 앞으로 걸어감).
추가: `logs/eval_videos/shuang_gap2048/pyramid_stairs_inv_d0.5_off/` (계단 앞에서 돌아섬),
`logs/eval_videos/shuang_gap2048/wave_d0_off/`.

## 4. 확실하지 않은 것

- **Shuang이 계단을 못 올라가는 이유**는 모릅니다. 영상에서는 첫 계단까지 가서 돌아서고, 구덩이 바닥을
  돌아다닙니다. 넘어지지는 않습니다. 학습 지형에 올라가는 계단도 있었기 때문에 이상합니다.
- **Shuang의 wave d = 0 (거의 평지)이 0.38**인 이유도 모릅니다 (d = 0.25는 1.00). 실패는 주로 무릎이 땅에 닿는
  접촉입니다. 이 평가는 무릎 접촉을 실패로 치는데, Shuang의 학습 task에는 그 규칙이 없습니다. 영상 1개에서는
  재현되지 않았습니다.
- 성공 기준을 바꿔서 **어제의 결과와 직접 비교하면 안 됩니다** (특히 Shuang의 stepping_stones d = 0: 옛 기준
  0.97, 새 기준 0.67 — 옆으로 걸어 나가는 로봇이 이제 실패/정지로 셈).
- 두 모델 모두 같은 평가 환경(AME task의 종료 규칙)을 썼습니다. Shuang에게는 학습 때보다 엄격합니다.

## 5. 문제가 있었던 것

- `stone_lattice` d = 0은 돌이 아주 많아서 run 하나가 **RAM 약 24 GB**를 씁니다. 두 모델을 동시에 돌렸더니
  메모리가 부족해서 run 1개가 실패했습니다. 그 뒤로는 **한 번에 한 모델씩** 돌렸고, 실패한 run은 다시 돌려서
  성공했습니다. 최종: 모델당 126 run 모두 정상 (총 252).
- 메모리 부족 때문에 Claude Code가 감시용 백그라운드 명령을 두 번 멈췄습니다. 평가 run 자체는 영향 없었습니다.

## 6. 내일 할 명령 (gap_1024)

먼저 2080 Ti의 run 폴더를 `checkpoints/gap_1024/`에 복사하세요. 그다음 repo 루트에서 (다른 평가와 동시에 돌리지
마세요, 브라우저도 가능하면 닫기):

```bash
nohup systemd-inhibit --what=sleep:idle --why=eval bash -c 'C=$(ls checkpoints/gap_1024/model_*.pt | sort -V | tail -1); echo "checkpoint: $C"; bash scripts/run_policy_matrix.sh gap_1024 "$C"; uv run python scripts/aggregate_matrix.py --models baseline shuang_gap2048 gap_1024' > logs/eval_results/matrix_gap_1024.log 2>&1 &
```

- 가장 큰 번호의 `model_*.pt`를 자동으로 고릅니다. 예상 시간 약 2.5-3시간.
- 끝나면 세 모델이 들어간 CSV와 그래프가 `logs/eval_results/matrix_plots/`에 다시 만들어집니다.
- 진행 확인: `tail logs/eval_results/matrix_gap_1024.log`
