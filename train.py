import os, random, math, numpy as np
from functools import partial
import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from strike5_environment import Strike5Env
from metrics_callback import MetricsCallback

CLEAR_2_REWARD = 1
CLEAR_3_REWARD = 20
CLEAR_4_REWARD = 50
CLEAR_5_REWARD = 100
REPEAT_MOVE_REWARD = -10
VALID_MOVE_REWARD = -0.1
INVALID_MOVE_REWARD = -10

SCALE_REWARDS = False
CUSTOM_SPAWN_RANGE = (3, 3)
PROBABILITY_OF_REGULAR_SPAWN = 0

END_GAME_BOARD_PERCENTAGE = 0.95
END_GAME_NUM_VALID_MOVES = math.inf
END_GAME_NUM_REPEATED_MOVES = 20
END_GAME_NUM_ATTEMPTED_MOVES = 250

LEARNING_RATE = 0.0005
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.01
VALUE_FUNCTION_COEFFICIENT = 0.5
MAX_GRADIENT_NORM = 0.5

RESUME_TRAINING_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "./logs_sb3/large_cnn_2.zip"
SAVE_FREQUENCY = 100000
TOTAL_TIMESTEPS = 5000000
NUM_ENVIRONMENTS = 2

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 256):
        total_concat_size = cnn_output_dim + 32
        super().__init__(observation_space, features_dim=total_concat_size)

        cnn_space = observation_space.spaces["cnn_features"]
        cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_cnn_input = torch.as_tensor(cnn_space.sample()[None]).float()
            n_flatten = cnn_extractor(sample_cnn_input.permute(0, 3, 1, 2)).shape[1]

        cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.LayerNorm(cnn_output_dim),
            nn.ReLU()
        )
        self.cnn = nn.Sequential(cnn_extractor, cnn_linear)

        vector_space = observation_space.spaces["vector_features"]
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_input = observations["cnn_features"].permute(0, 3, 1, 2)
        cnn_features = self.cnn(cnn_input)
        
        vector_features = self.vector_mlp(observations["vector_features"])
        
        return torch.cat([cnn_features, vector_features], dim=1)

def make_env(rank, seed=69420):
    def _init():
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        env = Strike5Env(
            clear_2_reward=CLEAR_2_REWARD,
            clear_3_reward=CLEAR_3_REWARD,
            clear_4_reward=CLEAR_4_REWARD,
            clear_5_reward=CLEAR_5_REWARD,
            repeat_move_reward=REPEAT_MOVE_REWARD,
            valid_move_reward=VALID_MOVE_REWARD,
            invalid_move_reward=INVALID_MOVE_REWARD,
            end_game_board_percentage=END_GAME_BOARD_PERCENTAGE,
            end_game_num_valid_moves=END_GAME_NUM_VALID_MOVES,
            end_game_num_repeated_moves=END_GAME_NUM_REPEATED_MOVES,
            end_game_num_attempted_moves=END_GAME_NUM_ATTEMPTED_MOVES,
            custom_spawn_range=CUSTOM_SPAWN_RANGE,
            probability_of_regular_spawn=PROBABILITY_OF_REGULAR_SPAWN,
            scale_rewards=SCALE_REWARDS
        )
        return env
    return _init

def main():
    log_dir = "./logs_sb3/"
    os.makedirs(log_dir, exist_ok=True)
    
    vec_env = DummyVecEnv([make_env(i, seed=42) for i in range(NUM_ENVIRONMENTS)])

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose = 1,
        device = "cpu",
        tensorboard_log = log_dir,
        batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        n_steps = N_STEPS,
        n_epochs = N_EPOCHS,
        gamma = GAMMA,
        gae_lambda = GAE_LAMBDA,
        clip_range = CLIP_RANGE,
        ent_coef = ENTROPY_COEFFICIENT,
        vf_coef = VALUE_FUNCTION_COEFFICIENT,
        max_grad_norm = MAX_GRADIENT_NORM,
    )

    if RESUME_TRAINING_FROM_CHECKPOINT and os.path.isfile(CHECKPOINT_PATH):
        old_model = MaskablePPO.load(CHECKPOINT_PATH, env=None)
        model.policy.load_state_dict(old_model.policy.state_dict())
        model.set_env(vec_env)
        print("Resuming training from " + CHECKPOINT_PATH)
    else:
        print("Training from scratch")

    checkpoint_cb = CheckpointCallback(
        save_freq = SAVE_FREQUENCY // NUM_ENVIRONMENTS,
        save_path = log_dir,
        name_prefix = "strike5_ppo",
    )
    metrics_cb = MetricsCallback()

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback = [checkpoint_cb, metrics_cb],
        tb_log_name = "strike5_run",
        reset_num_timesteps = True
    )

    model.save(os.path.join(log_dir, "strike5_ppo_final"))

if __name__ == "__main__":
    main()