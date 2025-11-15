# -*- coding: utf-8 -*-
"""
AvoidBlurp-Normal-v0 환경에서 DQN 학습 스크립트 (PyTorch 리팩터링 버전)

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

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ====================================================
# 0. Device 설정
# ====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ====================================================
# 1. 환경 빌더
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


# ====================================================
# 2. Q-network (PyTorch)
# ====================================================

class QNetwork(nn.Module):
    """
    입력: (B, 1, 84, 84)  [그레이스케일 후 리사이즈]
    출력: (B, n_actions)  [각 행동의 Q값]
    """

    def __init__(self, in_channels: int, n_actions: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Keras의 GlobalAveragePooling2D와 동일
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),           # (B, 128*1*1) -> (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        # Keras의 HeNormal/GlorotNormal에 대응하는 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


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
    """에피소드 수 기준 epsilon 값을 piecewise로 계산 (기존 Keras 버전 그대로)."""
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
    """
    (H, W, C) uint8 관측을
      -> float32 [0,1]
      -> (1, 1, 84, 84) 그레이스케일로 변환해 numpy로 반환.
    (학습 시에만 torch.Tensor로 변환해서 GPU로 보냄)
    """
    # numpy -> torch (CPU)
    x = torch.from_numpy(obs).float() / 255.0          # (H, W, C)
    x = x.permute(2, 0, 1).unsqueeze(0)               # (1, C, H, W)

    # RGB -> Gray
    r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
    x = 0.2989 * r + 0.5870 * g + 0.1140 * b          # (1, 1, H, W)

    # Resize to 84x84 (bilinear)
    x = F.interpolate(x, size=(84, 84), mode="bilinear", align_corners=False)
    return x.numpy()  # (1, 1, 84, 84)


class ReplayBuffer:
    """단순 균등 샘플링 replay buffer (state는 전처리된 (1,1,84,84) numpy 배열로 저장)."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.buf = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done) -> None:
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch: int):
        assert len(self.buf) >= batch
        idx = self.rng.choice(len(self.buf), size=batch, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (
            np.concatenate(s, axis=0),                   # (B, 1, 84, 84)
            np.array(a, dtype=np.int64),                # (B,)
            np.array(r, dtype=np.float32),              # (B,)
            np.concatenate(s2, axis=0),                  # (B, 1, 84, 84)
            np.array(d, dtype=np.bool_),                # (B,)
        )

    def __len__(self) -> int:
        return len(self.buf)


# ====================================================
# 5. 학습 루프 (PyTorch)
# ====================================================

def train_dqn(env: gym.Env, model: QNetwork, cfg: TrainConfig):
    """DQN 학습 메인 루프 (Keras 버전 로직을 그대로 반영)."""

    # Target network 초기화
    target_model = QNetwork(in_channels=1, n_actions=env.action_space.n, seed=cfg.seed).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.SmoothL1Loss()   # Huber loss

    replay = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    reward_history = []
    update_steps = 0

    pbar = tqdm(range(cfg.episodes), desc="Episode")

    model.train()

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
        state = preprocess(obs)  # (1,1,84,84) numpy
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
                action = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).to(DEVICE)  # (1,1,84,84)
                    q_values = model(state_t)[0].cpu().numpy()    # (A,)
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

                # numpy -> torch Tensor (GPU)
                s_batch_t = torch.from_numpy(s_batch).to(DEVICE)           # (B,1,84,84)
                s2_batch_t = torch.from_numpy(s2_batch).to(DEVICE)         # (B,1,84,84)
                a_batch_t = torch.from_numpy(a_batch).to(DEVICE)           # (B,)
                r_batch_t = torch.from_numpy(r_batch).to(DEVICE)           # (B,)
                done_batch_t = torch.from_numpy(done_batch.astype(np.float32)).to(DEVICE)  # (B,)

                # Q(s,a)와 타깃 계산
                q_curr = model(s_batch_t)                   # (B, A)
                with torch.no_grad():
                    q_next = target_model(s2_batch_t)       # (B, A)
                    max_next_q, _ = torch.max(q_next, dim=1)  # (B,)

                    target_q = r_batch_t + cfg.gamma * max_next_q * (1.0 - done_batch_t)

                # 선택한 행동에 대한 Q(s,a)만 추출
                q_curr_selected = q_curr.gather(1, a_batch_t.unsqueeze(1)).squeeze(1)  # (B,)

                loss = loss_fn(q_curr_selected, target_q)

                optimizer.zero_grad()
                loss.backward()

                # gradient clipping (global norm)
                if cfg.clipnorm is not None and cfg.clipnorm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clipnorm)

                optimizer.step()

                ep_loss += float(loss.item())
                update_steps += 1

                # --- Target Network Sync ---
                if update_steps % cfg.target_update == 0:
                    target_model.load_state_dict(model.state_dict())
                    target_model.eval()

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

    # 재현성 (완전 고정까지는 아니지만 기본 세팅)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(render_mode="rgb_array")

    # 관측 크기 확인 (디버그 용)
    obs, _ = env.reset()
    print(f"[INFO] Raw obs shape: {obs.shape}")  # (750, 600, 3) 예상

    n_actions = env.action_space.n

    # PyTorch Q-network 생성 (입력 채널=1: 그레이스케일)
    model = QNetwork(in_channels=1, n_actions=n_actions, seed=cfg.seed).to(DEVICE)
    print(model)

    trained_model, reward_history = train_dqn(env, model, cfg)

    # PyTorch state_dict 형태로 저장
    torch.save(trained_model.state_dict(), "avoidblurp_dqn_basic_reward.pt")
    env.close()
    print("[INFO] Training finished. Model saved to avoidblurp_dqn_basic_reward_v2.pt")


if __name__ == "__main__":
    main()
