# 방향을 고려한 커리큘럼 승급 규칙 (`rule="progress"`)

브랜치 `progress-rule` (upstream/main 49a85ca 기준). 기존 태스크의 학습 동작은 바꾸지 않음.

## 왜

`terrain_levels_floor`의 `rule="tracking"`은 타임아웃으로 끝난 에피소드에서
`sum(pelvis_speed) >= 0.6 * sum(command_speed)`이면 승급시킨다. `pelvis_speed`는 `|v_xy|`라서
방향을 보지 않는다. 계단 구덩이 안에서 빙빙 돌기만 해도 속도는 나오므로 승급된다.

## 무엇을 바꿨나

- `src/ame/mdp/metrics.py`: 새 메트릭 `pelvis_progress` = `dot(v_xy, cmd_xy) / max(|cmd_xy|, 1e-6)`
  (m/s, `pelvis_speed`와 같은 yaw 프레임 골반 속도). 명령 방향으로 간 속도만 양수, 옆으로 가면 0,
  뒤로 가면 음수. `|cmd_xy| ≈ 0`이면 0을 반환한다 (이때 `command_speed`도 0이므로 기존과 같이 "추종함"으로 처리).
- `src/ame/mdp/curriculums.py`: `rule="progress"` 추가. `tracking`과 완전히 같고, 비교 대상만
  `pelvis_speed` 대신 `pelvis_progress`. `tracking`은 그대로.
- `src/ame/env_cfg.py`: `add_tracking_metrics`에 `pelvis_progress` 등록. `ame_gap_env_cfg`에 `rule` 인자 추가
  (기본값은 `GAP_CURRICULUM["rule"]` = `"tracking"`이라 기존 태스크는 동일).
- `src/ame/__init__.py`: 새 태스크 `Ame-Mjlab-Gap-Progress-Velocity-Rough-G1` 등록 (`rule="progress"` 외 동일).

## 주의

- `add_tracking_metrics`가 모든 gap 태스크에 쓰이므로, 기존 `Ame-Mjlab-Gap-Velocity-Rough-G1`에서도
  `pelvis_progress`가 로그에 하나 더 찍힌다. 로깅만 늘고 보상·커리큘럼·관측은 바뀌지 않는다.
- 같은 0.6 비율이라도 progress는 speed 이하이므로 (`v·ĉ <= |v|`) 승급이 더 엄격해진다.
  명령을 정확히 따르면 두 값은 같다. 아크(yaw 명령) 중에는 yaw 프레임 기준이라 명령 방향도 같이 돌기 때문에
  제대로 추종하는 로봇은 손해를 보지 않는다.

## 학습 명령 (2080Ti, gap_1024와 같은 설정)

`checkpoints/gap_1024/params/agent.yaml`: seed 42, max_iterations 10000, num_envs 1024, logger tensorboard.

    nohup uv run train Ame-Mjlab-Gap-Progress-Velocity-Rough-G1 --env.scene.num-envs 1024 \
      --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_1024 \
      --agent.logger tensorboard > train_gap_progress_1024.log 2>&1 &
