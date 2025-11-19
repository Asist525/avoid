import gymnasium as gym
import kymnasium as kym
import torch
import numpy as np

from avoid import QNetwork, preprocess   

MODEL_PATH = "/home/ubuntu/avoid/avoid/avoidblurp_dqn_basic_reward_hard.pt"


class AvoidBlurpDQNAgent(kym.Agent):
    def __init__(self, model_path, seed=42, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # action 수 얻기 위해 환경 잠깐 띄움
        env = gym.make(
            id="kymnasium/AvoidBlurp-Normal-v0",
            render_mode="rgb_array",
            bgm=False,
            obs_type="image",
        )
        n_actions = env.action_space.n
        env.close()

        # 모델 불러오기
        self.model = QNetwork(in_channels=1, n_actions=n_actions, seed=seed).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def act(self, observation, info: dict):
        state = preprocess(observation)
        state = torch.from_numpy(state).to(self.device)
        q_values = self.model(state)[0]
        return int(q_values.argmax().item())


    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        return cls(path)


if __name__ == "__main__":
    agent = AvoidBlurpDQNAgent(MODEL_PATH)

    kym.evaluate(
        env_id="kymnasium/AvoidBlurp-Normal-v0",
        agent=agent,
        render_mode="human",
        bgm=True,
        obs_type="image"
    )
