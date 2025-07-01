import os, random, math, numpy as np
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from strike5_environment import Strike5Env
from metrics_callback import MetricsCallback

CLEAR_BALL_REWARD = 10
REPEAT_MOVE_REWARD = -10
OCCUPIED_EMPTY_REWARD = -0.1  # Valid move
OCCUPIED_EMPTY_NO_PATH_REWARD = -0.1
OCCUPIED_OCCUPIED_REWARD = -0.1
EMPTY_EMPTY_REWARD = -0.1
EMPTY_OCCUPIED_REWARD = -0.1

SCALE_REWARDS = False  # False = disable
CUSTOM_SPAWN_RANGE = (3, 3)  # (3, 3) = disable
PROBABILITY_OF_REGULAR_SPAWN = 0  # 0 = disable
END_GAME_BOARD_PERCENTAGE = 2  # 2 = disable
END_GAME_NUM_VALID_MOVES = math.inf  # math.inf = disable
END_GAME_NUM_INVALID_MOVES = math.inf  # math.inf = disable
END_GAME_NUM_REPEATED_MOVES = math.inf  # math.inf = disable
END_GAME_NUM_ATTEMPTED_MOVES = math.inf  # math.inf = disable

LEARNING_RATE = 0.01  # 0.00001 to 0.003
N_STEPS = 512  # 256 to 2048
BATCH_SIZE = 64  # 32 to 256
N_EPOCHS = 8  # 3 to 10
GAMMA = 0.99  # 0.95 to 0.999
GAE_LAMBDA = 0.95  # 0.9 to 0.98
CLIP_RANGE = 0.2  # 0.1 to 0.3
ENTROPY_COEFFICIENT = 0.01  # 0 to 0.01
VALUE_FUNCTION_COEFFICIENT = 0.5  # 0.5 to 1
MAX_GRADIENT_NORM = 0.5  # 0.5 to 1

RESUME_TRAINING_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "./logs_sb3/3_20_clear_1.zip"

SAVE_FREQUENCY = 50000
TOTAL_TIMESTEPS = 2000000
NUM_ENVIRONMENTS = 8

def make_env(rank, seed=69420):
    def _init():
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        env = Strike5Env(clear_ball_reward=CLEAR_BALL_REWARD, repeat_move_reward=REPEAT_MOVE_REWARD, occupied_empty_reward=OCCUPIED_EMPTY_REWARD, occupied_empty_no_path_reward=OCCUPIED_EMPTY_NO_PATH_REWARD, occupied_occupied_reward=OCCUPIED_OCCUPIED_REWARD, empty_empty_reward=EMPTY_EMPTY_REWARD, empty_occupied_reward=EMPTY_OCCUPIED_REWARD, end_game_board_percentage=END_GAME_BOARD_PERCENTAGE, end_game_num_valid_moves=END_GAME_NUM_VALID_MOVES, end_game_num_invalid_moves=END_GAME_NUM_INVALID_MOVES, end_game_num_repeated_moves=END_GAME_NUM_REPEATED_MOVES, end_game_num_attempted_moves=END_GAME_NUM_ATTEMPTED_MOVES, custom_spawn_range=CUSTOM_SPAWN_RANGE, probability_of_regular_spawn=PROBABILITY_OF_REGULAR_SPAWN, scale_rewards=SCALE_REWARDS)
        return env
    return _init

def main():
    log_dir = "./logs_sb3/"
    os.makedirs("./logs_sb3/", exist_ok=True)
    raw_vec = DummyVecEnv([make_env(i, seed=42) for i in range(NUM_ENVIRONMENTS)])
    vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose = 1,
        device = "cpu",
        tensorboard_log = log_dir,
        batch_size = BATCH_SIZE,
        learning_rate = lambda progress_remaining: LEARNING_RATE * progress_remaining,
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
        old_model = PPO.load(CHECKPOINT_PATH, env=None)
        model.policy.load_state_dict(old_model.policy.state_dict())
        model.set_env(vec_env)
        print("Resuming training from " + CHECKPOINT_PATH)
    else: print("Training from scratch")

    checkpoint_cb = CheckpointCallback(
        save_freq = SAVE_FREQUENCY // NUM_ENVIRONMENTS,  # because each step call runs NUM_ENVIRONMENTS in parallel
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

if __name__ == "__main__": main()