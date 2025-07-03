import os, random, math, numpy as np
from functools import partial

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from strike5_environment import Strike5Env
from metrics_callback import MetricsCallback

CLEAR_BALL_REWARD = 25
REPEAT_MOVE_REWARD = -20
VALID_MOVE_REWARD = 0
INVALID_MOVE_REWARD = -20

SCALE_REWARDS = False
CUSTOM_SPAWN_RANGE = (3, 3)
PROBABILITY_OF_REGULAR_SPAWN = 0

END_GAME_BOARD_PERCENTAGE = 0.9
END_GAME_NUM_VALID_MOVES = math.inf
END_GAME_NUM_REPEATED_MOVES = math.inf
END_GAME_NUM_ATTEMPTED_MOVES = math.inf

LEARNING_RATE = 0.001
N_STEPS = 256
BATCH_SIZE = 64
N_EPOCHS = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.01
VALUE_FUNCTION_COEFFICIENT = 0.5
MAX_GRADIENT_NORM = 0.5

RESUME_TRAINING_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "./logs_sb3/mask_5.zip"
SAVE_FREQUENCY = 100000
TOTAL_TIMESTEPS = 1000000
NUM_ENVIRONMENTS = 8

def make_env(rank, seed=69420):
    def _init():
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        env = Strike5Env(
            clear_ball_reward=CLEAR_BALL_REWARD,
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
        env = ActionMasker(env, lambda env: env.action_masks())
        return env
    return _init

def main():
    log_dir = "./logs_sb3/"
    os.makedirs(log_dir, exist_ok=True)
    
    vec_env = DummyVecEnv([make_env(i, seed=42) for i in range(NUM_ENVIRONMENTS)])

    model = MaskablePPO(
        "MultiInputPolicy",
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