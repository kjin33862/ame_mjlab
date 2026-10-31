# 월드 좌표 속도 명령 (`rel_world_envs = 0.5`)

브랜치 `world-cmd` (`progress-rule` `cd93ec0` 기준). 기존 태스크는 바꾸지 않음.

## 왜

`Ame-Mjlab-Gap-Progress-Velocity-Rough-G1`의 명령은 전부 몸 좌표 twist이고, 약 3/4에 yaw 명령이 들어 있다.
몸 좌표에서는 로봇이 돌면 명령 방향도 같이 돈다. 그래서 계단 구덩이 안에서 빙빙 도는 것도 "명령을 따르는 것"이 되고,
`pelvis_progress`로 바꿔도 이 구멍은 남는다. 학습된 정책은 계단을 피했다 (gap_progress_1024: 구덩이 계단 d ≥ 0.25 성공 ≤ 0.07).

월드 좌표 명령은 선속도 방향이 월드에 고정되어 있다. 돌면서 걸어도 같은 방향으로 계속 가야 하므로,
도는 것으로는 진행(`pelvis_progress`)이 거의 0이 된다.

## 무엇을 바꿨나

- `src/ame/env_cfg.py`: `ame_gap_env_cfg`에 `rel_world_envs` 인자 추가 (기본 0.0 = mjlab 기본값과 같음).
- `src/ame/__init__.py`: 새 태스크 `Ame-Mjlab-Gap-ProgressWorld-Velocity-Rough-G1` 등록
  (`rule="progress"`, `rel_world_envs=0.5`). 나머지 명령 설정은 그대로 (heading 끔, 정지 0.05, 전진 전용 0.2,
  범위 vx −0.3–1.5, vy ±0.5, wz ±1.0, 3–8 s마다 재샘플).
- 기존 5개 태스크의 설정(학습, play)은 변경 전과 똑같음을 확인함 (함수 주소 외 repr 동일).

## 동작 확인 (mjlab `velocity_command.py`, 256 env 실행으로 확인)

- 매 step `command_manager.compute`에서 월드 벡터를 현재 heading으로 몸 좌표로 돌린다.
  `pelvis_progress`(와 보상, 관측)는 이 몸 좌표 명령을 쓴다. 한 policy step(0.02 s) 늦은 heading 기준이지만
  정책이 본 명령과 같다. 회전은 내적을 보존하므로 `pelvis_progress`는 사실상 월드 고정 방향으로의 진행 속도이다.
- 정지 env (5 %)는 회전 뒤에 0으로 덮어쓰므로 그냥 정지다.
- yaw 명령(`ang_vel_z`)은 회전하지 않고 그대로 남는다. 월드 env의 약 75 %가 yaw 명령을 받는다:
  돌면서 월드 고정 방향으로 가야 한다.
- 월드 ∩ 전진 전용 env (약 10 %): 월드 벡터가 전진 보정 전에 복사되므로 xy는 원래 샘플(뒤/옆 가능), wz = 0.

## 주의 (바꾸지 않음)

- 월드 벡터는 몸 좌표 범위에서 뽑은 값을 **월드 축**으로 해석한다. 시작 방향이 0/90/180/270°이므로 월드 env의
  3/4는 시작부터 몸 좌표 범위 밖의 명령을 받는다 (예: 180°에서 vx = −1.5 m/s 뒤로, 90°에서 vy = ±1.5 m/s 옆으로).
  실행에서 본 몸 좌표 명령 범위: vx, vy 모두 약 ±1.5. 추종이 어려워 강등이 늘 수 있다.
- 결과는 아직 없음. `notes/experiments.md`에 기록할 것.
