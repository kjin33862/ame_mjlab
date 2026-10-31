# heading 명령 다시 켜기 (`Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1`)

브랜치 `world-cmd` (`progress-rule` `cd93ec0` 기준). 기존 태스크는 바꾸지 않음.

## 왜

`Ame-Mjlab-Gap-Progress-Velocity-Rough-G1`의 명령은 전부 몸 좌표 twist이고, 약 3/4에 무작위 yaw 명령이 들어 있다.
몸 좌표에서는 로봇이 돌면 명령 방향도 같이 돈다. 그래서 계단 구덩이 안에서 빙빙 도는 것도 "명령을 따르는 것"이 되고,
`pelvis_progress`로 바꿔도 이 구멍은 남는다. 학습된 정책은 계단을 피했다 (gap_progress_1024: 구덩이 계단 d ≥ 0.25 성공 ≤ 0.07).

heading 명령은 legged_gym, Isaac Lab, mjlab 자체 velocity 태스크의 표준 설정이다 (Shuang이 껐음).
heading env는 무작위 yaw 속도 대신 **월드 기준 목표 방향**으로 돌아서고, 맞춘 뒤에는 wz ≈ 0으로 곧게 걷는다.
그래서 무작위 yaw 명령 비율이 약 76 %에서 약 53 %로 줄고, 30 %의 명령은 한 방향으로 곧게 가는 것을 요구한다.

## 월드 좌표 명령 버전(179cc5b)을 버린 이유

- 월드 env도 무작위 yaw 명령을 그대로 받았다 (약 75 %). 즉 "월드 한 방향으로 걸으면서 계속 돌아라"는 명령이다.
  부자연스럽고, 실제 planner가 보낼 명령이 아니다.
- 월드 벡터를 몸 좌표 범위에서 뽑아 월드 축으로 해석하므로, 시작 방향(0/90/180/270°)에 따라 월드 env의 3/4가
  범위 밖 명령을 받았다 (예: 뒤로 1.5 m/s, 옆으로 1.5 m/s).
- 그 태스크(`...-ProgressWorld-...`)와 로컬 태그 `run/gap_progress_world_1024`는 삭제했다 (push한 적 없음).
  커밋 179cc5b는 기록으로 남겨 둠.

## 무엇을 바꿨나

- `src/ame/env_cfg.py`: 새 mutator `apply_heading_commands` (mjlab velocity 태스크 기본값:
  `heading_command=True`, `rel_heading_envs=0.3`, `heading_control_stiffness=0.5`, `ranges.heading=(-π, π)`).
  `ame_gap_env_cfg`의 `rel_world_envs` 인자를 `heading_command` 인자(기본 False)로 바꿈. `rel_world_envs`는 0 (mjlab 기본값).
- `src/ame/__init__.py`: 태스크 이름을 `Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1`로 바꾸고
  `rule="progress", heading_command=True`로 등록.
- Progress 태스크와의 실제 차이는 `heading_command`, `rel_heading_envs`, `ranges.heading` 세 개
  (stiffness는 원래 0.5). 정지 0.05, 전진 전용 0.2, 범위 vx −0.3–1.5, vy ±0.5, wz ±1.0, 3–8 s 재샘플은 그대로.
- 기존 5개 태스크 설정(학습, play)은 `progress-rule`과 똑같음을 확인함.

## 동작 (mjlab `velocity_command.py`, 1024 env 실행으로 확인)

- heading env의 wz = clip(0.5 × wrap(목표 방향 − 현재 heading), −1, 1), **매 step** 다시 계산. 샘플한 wz는 덮어씀.
  목표 방향은 재샘플마다 U(−π, π) (월드 기준). 오차가 2 rad 넘으면 1 rad/s로 포화, 시정수 약 2 s.
- vx, vy는 그대로 몸 좌표 범위에서 샘플 (vx −0.3–1.5, vy ±0.5).
- 정지 env: 마지막에 명령 전체를 0으로 덮으므로 heading env여도 그냥 정지.
- 전진 전용 env: vx ≥ 0.3, vy = 0. heading env이기도 하면 wz는 heading 제어가 덮어씀 (돌아서면서 앞으로).
  heading이 아니면 wz = 0.
- 명령 비율 (독립 추첨): 정지 5 %, heading+전진 5.7 %, heading만 22.8 %, 전진만 13.3 %, 무작위 twist 53.2 %.
  실행에서 잰 값: 4.8 / 5.9 / 22.3 / 13.6 / 53.4 %.

## 주의

- 무작위 twist 53 %는 여전히 몸 좌표라서, 그 부분에서는 도는 것이 진행으로 셈해지는 구멍이 남는다.
- 결과는 아직 없음. `notes/experiments.md`에 기록할 것.
