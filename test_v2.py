import gymnasium as gym
import kymnasium as kym
import torch
import numpy as np

from avoid_v2 import QNetwork, preprocess   # 학습과 동일한 함수 import

MODEL_PATH = "/home/ubuntu/avoid/avoid/avoidblurp_dqn_basic_reward_framestack_v2.pt"
FRAME_STACK = 4  # 학습과 반드시 동일해야 함


class AvoidBlurpDQNAgent(kym.Agent):
    def __init__(self, model_path, seed=42, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # 1) 액션 수 파악
        # ----------------------------------------------------
        tmp_env = gym.make(
            id="kymnasium/AvoidBlurp-Normal-v0",
            render_mode="rgb_array",
            bgm=False,
            obs_type="image",
        )
        self.n_actions = tmp_env.action_space.n
        tmp_env.close()

        # ----------------------------------------------------
        # 2) 모델 로드 (frame_stack=4)
        # ----------------------------------------------------
        self.model = QNetwork(
            in_channels=FRAME_STACK,
            n_actions=self.n_actions,
            seed=seed
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ----------------------------------------------------
        # 3) frame stack 초기화 공간
        # ----------------------------------------------------
        self.frame_stack = None

    # --------------------------------------------------------
    # Evaluation init hook (에피소드 시작 시 자동 호출)
    # --------------------------------------------------------
    def reset(self):
        self.frame_stack = None

    # --------------------------------------------------------
    # Act 함수
    # --------------------------------------------------------
    @torch.no_grad()
    def act(self, observation, info: dict):

        # 1) 새 프레임 전처리 → (84,84) uint8
        frame = preprocess(observation)

        # 2) frame stack 초기화
        if self.frame_stack is None:
            self.frame_stack = np.stack([frame] * FRAME_STACK, axis=0)
        else:
            # 오른쪽으로 shift하고 마지막에 새 프레임 append
            self.frame_stack = np.concatenate(
                [self.frame_stack[1:], frame[None, ...]], axis=0
            )

        # 3) 모델 입력 준비 (float32 / 255)
        state = self.frame_stack.astype(np.float32) / 255.0
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # 4) Q-value 계산
        q_values = self.model(state_t)[0]

        # 5) greedy action
        return int(torch.argmax(q_values).item())

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        return cls(path)


# --------------------------------------------------------
# 실행
# --------------------------------------------------------
if __name__ == "__main__":
    agent = AvoidBlurpDQNAgent(MODEL_PATH)

    kym.evaluate(
        env_id="kymnasium/AvoidBlurp-Normal-v0",
        agent=agent,
        render_mode="human",
        bgm=True,
        obs_type="image"
    )
