# -*- coding: utf-8 -*-
"""
AvoidBlurp-Normal-v0 환경에서 DQN 학습 스크립트 (리팩터링 버전)

- obs_type="image" (750x600x3 RGB)
- Q-network: CNN + GlobalAveragePooling + Dense
- 보상 설계:
    * 매 step 살아있으면 +ALIVE_REWARD
    * 2분 완주(terminated=True) 시 SUCCESS_BONUS 추가
    * 충돌(truncated=True) 시 DEATH_PENALTY 추가
"""

import kymnasium                      # env 등록용 (직접 쓰진 않지만 필요)
import gymnasium as gym
import numpy as np
from collections import deque
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm


# ====================================================
# 1. Custom Layer 정의
# ====================================================

@keras.utils.register_keras_serializable()
class GrayScaleLayer(keras.layers.Layer):
    """RGB 이미지를 단일 채널 그레이스케일로 변환하는 래퍼 레이어."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.image.rgb_to_grayscale(inputs)


# ====================================================
# 2. 환경 및 네트워크 빌더
# ====================================================

def make_env(render_mode: str = "rgb_array") -> gym.Env:
    """AvoidBlurp 환경 생성."""
    env = gym.make(
        id="kymnasium/AvoidBlurp-Normal-v0",
        render_mode=render_mode,  # "rgb_array" (학습용)
        bgm=False,
        obs_type="image",
    )
    return env


def build_q_network(input_shape, n_actions: int, seed: int = 42) -> keras.Model:
    """이미지 관측을 입력으로 받는 Q-network 생성."""
    he_init = keras.initializers.HeNormal(seed=seed)
    glorot_init = keras.initializers.GlorotNormal(seed=seed)

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=input_shape),        # (750, 600, 3)
            keras.layers.Resizing(height=84, width=84),   # 다운샘플링
            GrayScaleLayer(),                             # (84, 84, 1)
            keras.layers.Rescaling(scale=1.0 / 255.0),    # [0,255] → [0,1]

            keras.layers.Conv2D(
                filters=32, kernel_size=8, padding="same",
                activation="relu", kernel_initializer=he_init,
            ),
            keras.layers.MaxPool2D(pool_size=2),

            keras.layers.Conv2D(
                filters=64, kernel_size=4, padding="same",
                activation="relu", kernel_initializer=he_init,
            ),
            keras.layers.Conv2D(
                filters=64, kernel_size=4, padding="same",
                activation="relu", kernel_initializer=he_init,
            ),
            keras.layers.MaxPool2D(pool_size=2),

            keras.layers.Conv2D(
                filters=128, kernel_size=3, padding="same",
                activation="relu", kernel_initializer=he_init,
            ),
            keras.layers.Conv2D(
                filters=128, kernel_size=3, padding="same",
                activation="relu", kernel_initializer=he_init,
            ),
            keras.layers.MaxPool2D(pool_size=2),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(
                units=64, activation="relu", kernel_initializer=he_init
            ),
            keras.layers.Dense(
                units=32, activation="relu", kernel_initializer=he_init
            ),
            keras.layers.Dense(
                units=n_actions,
                activation="linear",
                kernel_initializer=glorot_init,
            ),
        ]
    )
    return model


# ====================================================
# 3. 설정 값 dataclass
# ====================================================

@dataclass
class TrainConfig:
    # 에피소드 관련
    episodes: int = 8000

    # DQN 관련
    gamma: float = 0.99
    batch_size: int = 32
    train_start: int = 1000
    target_update: int = 1000

    # Epsilon 스케줄
    eps_max: float = 1.0
    eps_min: float = 0.1
    warmup_ratio: float = 0.06     # 6%: epsilon = 1.0
    decay_ratio: float = 0.5       # 50%: 1.0 → 0.1
    final_ratio: float = 0.04      # 마지막 4%: 0.1 → 0.0 (hold는 나머지)

    # 보상 설계
    alive_reward: float = 0.1      # step당 생존 보상
    death_penalty: float = -10.0   # 충돌 시 패널티 (truncated=True)
    success_bonus: float = 10.0    # 2분 완주 보너스 (terminated=True)

    # 최적화
    learning_rate: float = 2.5e-4
    clipnorm: float = 1.0

    # 기타
    replay_capacity: int = 50_000
    seed: int = 42


# ====================================================
# 4. Epsilon 스케줄 / 전처리 / ReplayBuffer
# ====================================================

def calc_epsilon(
    ep: int,
    max_ep: int,
    eps_max: float,
    eps_min: float,
    warmup_ratio: float,
    decay_ratio: float,
    final_ratio: float,
) -> float:
    """에피소드 수 기준 epsilon 값을 piecewise로 계산."""
    warmup = int(max_ep * warmup_ratio)
    decay = int(max_ep * decay_ratio)
    final = int(max_ep * final_ratio)

    hold_start = warmup + decay
    hold_end = max_ep - final

    ep = min(max(ep, 0), max_ep)

    # 1) warmup (ε = 1.0)
    if ep < warmup:
        return eps_max

    # 2) decay (1.0 → 0.1)
    elif ep < warmup + decay:
        ratio = (ep - warmup) / max(decay, 1)
        ratio = min(ratio, 1.0)
        return eps_max - (eps_max - eps_min) * ratio

    # 3) hold (ε = 0.1)
    elif ep < hold_end:
        return eps_min

    # 4) final decay (0.1 → 0.0)
    else:
        if final <= 0:
            return 0.0
        ratio = (ep - hold_end) / final
        ratio = min(ratio, 1.0)
        return eps_min * (1.0 - ratio)


def preprocess(obs: np.ndarray) -> np.ndarray:
    """(H, W, C) 관측을 float32 + 배치 차원 추가."""
    return obs.astype(np.float32)[None, ...]  # (1, H, W, C)


class ReplayBuffer:
    """단순 균등 샘플링 replay buffer."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.buf = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done) -> None:
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch: int):
        idx = self.rng.choice(len(self.buf), size=batch, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (
            np.concatenate(s, axis=0),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.concatenate(s2, axis=0),
            np.array(d, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ====================================================
# 5. 학습 루프
# ====================================================

def train_dqn(env: gym.Env, model: keras.Model, cfg: TrainConfig):
    """DQN 학습 메인 루프."""
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    optimizer = keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
        clipnorm=cfg.clipnorm,
    )
    loss_fn = keras.losses.Huber(delta=1.0)

    replay = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    reward_history = []
    update_steps = 0

    pbar = tqdm(range(cfg.episodes), desc="Episode")

    for ep in pbar:
        # --- epsilon 계산 ---
        epsilon = calc_epsilon(
            ep=ep,
            max_ep=cfg.episodes,
            eps_max=cfg.eps_max,
            eps_min=cfg.eps_min,
            warmup_ratio=cfg.warmup_ratio,
            decay_ratio=cfg.decay_ratio,
            final_ratio=cfg.final_ratio,
        )

        obs, _ = env.reset()
        state = preprocess(obs)
        done = False

        ep_reward = 0.0
        ep_loss = 0.0
        steps = 0

        # ==========================
        #   Episode loop
        # ==========================
        while not done:
            # --- ε-greedy policy ---
            if rng.random() < epsilon:
                action = rng.integers(env.action_space.n)
            else:
                q_values = model(state).numpy()[0]  # (A,)
                action = int(np.argmax(q_values))

            # --- 환경 step ---
            next_obs, _, terminated, truncated, info = env.step(action)
            next_state = preprocess(next_obs)

            # AvoidBlurp 규칙:
            # - 충돌: terminated=False, truncated=True
            # - 2분 완주: terminated=True, truncated=False
            done = bool(terminated or truncated)

            # --- 보상 설계 ---
            reward = cfg.alive_reward  # 살아있는 한 기본 보상

            if terminated:
                # 2분 완주 성공
                reward += cfg.success_bonus
            elif truncated:
                # 충돌로 인한 종료
                reward += cfg.death_penalty

            # Replay에 transition 저장
            replay.add(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            steps += 1

            # --- DQN Update ---
            if len(replay) >= cfg.train_start:
                (
                    s_batch,
                    a_batch,
                    r_batch,
                    s2_batch,
                    done_batch,
                ) = replay.sample(cfg.batch_size)

                with tf.GradientTape() as tape:
                    q_curr = model(s_batch)            # (B, A)
                    q_next = target_model(s2_batch)    # (B, A)

                    max_next_q = tf.reduce_max(q_next, axis=1)  # (B,)

                    # done=True인 transition은 bootstrap 제거
                    done_f = tf.cast(done_batch, tf.float32)
                    target_q = r_batch + cfg.gamma * max_next_q * (1.0 - done_f)

                    # 선택된 행동의 Q(s,a)만 추출
                    mask = tf.one_hot(a_batch, env.action_space.n)
                    q_curr_selected = tf.reduce_sum(q_curr * mask, axis=1)

                    loss = loss_fn(target_q, q_curr_selected)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                ep_loss += float(loss)
                update_steps += 1

                # --- Target Network Sync ---
                if update_steps % cfg.target_update == 0:
                    target_model.set_weights(model.get_weights())

        # --- Episode logging ---
        avg_reward = ep_reward / max(steps, 1)
        avg_loss = ep_loss / max(steps, 1)

        reward_history.append(avg_reward)
        if len(reward_history) > 100:
            reward_history.pop(0)

        pbar.set_postfix(
            eps=f"{epsilon:.3f}",
            reward=f"{avg_reward:.3f}",
            loss=f"{avg_loss:.5f}",
            steps=steps,
        )

    return model, reward_history


# ====================================================
# 6. 실행 진입점
# ====================================================

def main():
    cfg = TrainConfig()
    env = make_env(render_mode="rgb_array")

    # 관측 크기 확인 후 네트워크 생성
    obs, _ = env.reset()
    input_shape = obs.shape          # (750, 600, 3)
    n_actions = env.action_space.n

    model = build_q_network(input_shape, n_actions, seed=cfg.seed)
    model.summary()

    trained_model, reward_history = train_dqn(env, model, cfg)

    # 필요하면 모델 저장
    trained_model.save("avoidblurp_dqn_basic_reward.h5")
    env.close()


if __name__ == "__main__":
    main()
