````markdown
# AvoidBlurp DQN Experiments

`kymnasium/AvoidBlurp-Normal-v0` 환경에서 DQN 계열 알고리즘을 단계적으로 확장·비교하는 프로젝트이다.  
V1 ~ V5까지는 실제 구현/실험 라인, V6 ~ V10은 향후 확장 로드맵이다.

---

## 1. 환경 및 문제 정의

- **환경**: `kymnasium/AvoidBlurp-Normal-v0`
- **관측**:  
  - `obs_type="image"`  
  - 원본: `(750, 600, 3)` RGB, `uint8`
  - 전처리:
    - 그레이스케일 변환
    - `84 x 84`로 리사이즈
    - V1: 1프레임, V2부터: 4프레임 frame stacking
- **액션 공간**: `Discrete(3)`
  - `0`: 정지
  - `1`: 왼쪽 이동
  - `2`: 오른쪽 이동  (실제 매핑은 env 정의에 따름)
- **에피소드 종료 규칙** (AvoidBlurp 규칙):
  - **충돌**: `terminated=False`, `truncated=True`
  - **2분 완주**: `terminated=True`, `truncated=False`
  - 코드에서는 `done = terminated or truncated`

### 보상 설계 (V1 ~ V4 공통, V5에서 변경 예정)

- 매 스텝 생존: `+0.1` (`alive_reward`)
- 2분 완주 성공: `+10.0` (`success_bonus`)
- 충돌로 종료: `-10.0` (`death_penalty`)

즉, 기본적으로 **“최대한 오래 살아남아라”**가 목표인 에이전트이다.

---

## 2. 코드 구조

리포지토리 루트 기준:

- `avoid.py`  
  - **V1: 기본 DQN (single frame)**  
  - PyTorch 기반 CNN + GAP + FC 구조  
  - `(1, 1, 84, 84)` 형태의 단일 프레임을 입력으로 사용
- `avoid_v2.py`  
  - **V2: DQN + frame stacking (4 frames)**  
  - 입력: `(4, 84, 84)` (최근 4프레임 gray stack)  
  - seed 고정, CPU↔GPU 병목 개선, 메모리 사용량 감소
- `test.py`  
  - 학습된 PyTorch 네트워크를 `kym.evaluate` 인터페이스에 맞게 감싼 **평가용 스크립트**
  - `AvoidBlurpDQNAgent`가 내부에서 모델을 로드하고, `act()`로 행동만 반환
  - **추가 학습 없음 (순수 평가/리플레이 전용)**

향후 버전(V3 이후)은 위 구조를 그대로 확장하는 형태로 설계한다.

---

## 3. 설치 및 실행

### 3.1. 환경 준비

```bash
python -m venv venv
source venv/bin/activate  # Windows WSL/Ubuntu 기준
pip install --upgrade pip

pip install torch gymnasium kymnasium pygame tqdm numpy
# 필요 시 CUDA 지원 torch는 공식 가이드에 따라 설치
````

### 3.2. V1 학습 (단일 프레임 DQN)

```bash
python avoid.py
```

* 예상 동작:

  * `Using device: cuda` (또는 cpu)
  * 원본 obs shape 출력: `(750, 600, 3)`
  * `Episode ... eps=..., reward=..., loss=..., steps=...` 형태의 tqdm 로그
* 모델 저장:

  * `avoidblurp_dqn_basic_reward.pt` (state_dict)

### 3.3. V2 학습 (frame stacking DQN)

```bash
python avoid_v2.py
```

* 주요 차이:

  * `Conv2d(4, 32, ...)` → 입력 채널 4 (frame 4장)
  * replay buffer에 `(frame_stack, 84, 84)`를 `uint8`로 저장
* 모델 저장:

  * `avoidblurp_dqn_basic_reward_framestack_v2.pt` (state_dict)

로그 예시:

```text
[INFO] Using device: cuda
[INFO] Raw obs shape: (750, 600, 3)
QNetwork(
  (features): Sequential(
    (0): Conv2d(4, 32, ...
Episode:   0%|▏ ... eps=1.000, loss=..., reward=..., steps=...
)
```

---

## 4. 평가 방법 (kym.evaluate)

`test.py`에서 `AvoidBlurpDQNAgent`가 `kym.Agent`를 상속하고,
`act()`, `save()`, `load()`를 구현해 `kym.evaluate`에 넘긴다.

### 4.1. V1 평가 예시

```python
# test.py 내부 (예시)
MODEL_PATH = "/home/ubuntu/avoid/avoid/avoidblurp_dqn_basic_reward.pt"

agent = AvoidBlurpDQNAgent(MODEL_PATH)

kym.evaluate(
    env_id="kymnasium/AvoidBlurp-Normal-v0",
    agent=agent,
    render_mode="human",
    bgm=True,
    obs_type="image",
)
```

* `AvoidBlurpDQNAgent.act(observation, info)`에서:

  * `observation`: `(750, 600, 3)` 이미지
  * `preprocess` → `(1, 1, 84, 84)` 텐서
  * `model(state)` → `argmax`로 행동 결정

### 4.2. V2 평가 예시

V2는 frame stacking을 사용하므로, 평가용 Agent도 동일하게 4프레임을 쌓아서 입력해야 한다.
구조는 동일하고, 다음만 다르다.

* `QNetwork(in_channels=4, ...)` 로 초기화
* 첫 4프레임을 쌓은 뒤, 새로운 프레임 들어올 때마다 shift + append

---

## 5. 버전 로드맵 (V1 ~ V10)

현재/계획 버전들을 정리하면 다음과 같다.

| Version | 알고리즘/구조                                         | 핵심 변경점                                                       | 상태    |
| ------- | ----------------------------------------------- | ------------------------------------------------------------ | ----- |
| **V1**  | DQN                                             | 단일 프레임 `(1, 84, 84)` 입력, vanilla DQN                         | 구현 완료 |
| **V2**  | DQN + Frame Stacking                            | 4프레임 stack, seed 고정, CPU↔GPU 병목 개선, uint8 replay             | 구현 완료 |
| **V3**  | **Double DQN + Frame**                          | 타깃 계산에 Double Q 도입 (online으로 argmax, target으로 value)         | 계획    |
| **V4**  | **Double DQN + Frame + PER**                    | Prioritized Experience Replay (α, β 도입)                      | 계획    |
| **V5**  | **Double DQN + Frame + PER + Re-Reward**        | 보상 재설계 (shaping/스케일 조정, 위험 회피 보상 등)                          | 계획    |
| **V6**  | **Dueling Double DQN + Frame (+ PER)**          | Dueling network (V(s) + A(s,a)) 구조 도입                        | 계획    |
| **V7**  | **n-step Dueling Double DQN + Frame**           | multi-step TD (n-step return) 적용                             | 계획    |
| **V8**  | **NoisyNet Exploration (+ ε 축소)**               | NoisyLinear로 탐험 개선, ε-greedy 최소화                             | 계획    |
| **V9**  | **Distributional DQN (C51/QR) + Dueling + PER** | Q 분포(원자 혹은 quantile) 예측, distributional loss 사용              | 계획    |
| **V10** | **Rainbow-lite 통합 모델**                          | Double + Dueling + PER + n-step + Noisy (+/− Distributional) | 계획    |

설명:

* **V1 → V2**: 관측 표현 개선 (frame stacking) + 구현 안정화
* **V2 → V3**: Q-value 과대추정 개선 (Double DQN)
* **V3 → V4**: 샘플 효율 개선 (PER)
* **V4 → V5**: 보상 구조 자체 수정 → 정책의 “행동 양식” 변경 실험
* **V6~V10**: DQN 계열 연구 흐름(Dueling, n-step, Noisy, Distributional, Rainbow)을 단계적으로 이식

---

## 6. 실험 프로토콜 (비교 실험 가이드)

### 6.1. V1 vs V2

* 학습:

  * 동일한 `episodes` 설정 (예: 2000 에피소드)
* 평가:

  * 각 버전 모델에 대해 `test.py`로 20 에피소드 평가
  * 지표:

    * 평균 생존 시간(초) 또는 step 수
    * 평균 episode reward
    * 2분 완주 성공률
    * 성능 분산(표준편차)

기대:

* V1: 약 60~80초 수준 생존
* V2: 같은 budget에서 더 높은 생존 시간/성공률 (frame stacking + 안정화 효과)

### 6.2. 향후 버전 실험

* **V3**: V2와 동일 환경·하이퍼에서 Double DQN만 켜고 성능 차이 비교
* **V4**: PER on/off에 따른 sample 효율 비교
* **V5**: 기존 보상 vs 새로운 보상에서 학습된 정책의 **행동 양식** 비교
* **V6~V10**: 각 기능 추가에 따라 학습 곡선과 policy quality 변화를 단계적으로 분석

---

## 7. TODO 정리

* [ ] V3: Double DQN 타깃 계산 코드 반영
* [ ] V2/V3 통합 평가용 Agent (frame stacking 버전) 정리
* [ ] V1~V3 실험 결과(생존 시간/성공률) 그래프/테이블 작성
* [ ] V4: PER 적용 ReplayBuffer 구현
* [ ] V5: Reward shaping 설계안 2~3개 정의 및 실험
* [ ] V6~V10: Dueling, n-step, Noisy, Distributional, Rainbow-lite 순차 구현 및 실험

---

## 8. 요약

* V1~V2에서 **표현/병목/안정성**을 먼저 잡고,
* V3~V4에서 **알고리즘적 개선(DDQN, PER)**을 도입한 뒤,
* V5에서 **Reward 설계**를 건드리고,
* V6~V10에서 **DQN 계열 주요 연구 아이디어들을 단계적으로 이식**하는 로드맵이다.

이 README는 구현/실험/보고서에서 “버전 간 차이와 발전 방향”을 설명하는 기준 문서 역할을 한다.

```
::contentReference[oaicite:0]{index=0}
```
