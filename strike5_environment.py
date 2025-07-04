import math, random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete
from strike5_engine import reset_board, apply_move, spawn_balls, GRID_SIZE, SPAWN_COUNT

class Strike5Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, clear_ball_reward, repeat_move_reward, valid_move_reward, invalid_move_reward, end_game_board_percentage=2, end_game_num_valid_moves=math.inf, end_game_num_repeated_moves=math.inf, end_game_num_attempted_moves=math.inf, custom_spawn_range=(3, 3), probability_of_regular_spawn=0, scale_rewards=False):
        super().__init__()
        self.clear_ball_reward = clear_ball_reward
        self.repeat_move_reward = repeat_move_reward
        self.valid_move_reward = valid_move_reward
        self.invalid_move_reward = invalid_move_reward
        self.end_game_board_percentage = end_game_board_percentage
        self.end_game_num_valid_moves = end_game_num_valid_moves
        self.end_game_num_repeated_moves = end_game_num_repeated_moves
        self.end_game_num_attempted_moves = end_game_num_attempted_moves
        self.custom_spawn_range = custom_spawn_range
        self.probability_of_regular_spawn = probability_of_regular_spawn
        self.scale_rewards = scale_rewards

        self.action_space = Discrete(GRID_SIZE**4)

        self.observation_space = Dict({
            "action_mask": Box(low=0, high=1, shape=(GRID_SIZE**4,), dtype=bool),
            "cnn_features": Box(low=0, high=7, shape=(GRID_SIZE, GRID_SIZE, 1), dtype=np.uint8),
            "vector_features": Box(low=0, high=7, shape=(SPAWN_COUNT,), dtype=np.uint8),
        })
        
        self.state = None
        self.num_attempted_moves = 0
        self.num_valid_moves = 0
        self.num_balls_on_valid = 0
        self.last_action = 0
        self.num_repeated_moves = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.num_attempted_moves = 0
        self.num_valid_moves = 0
        self.num_repeated_moves = 0
        self.num_balls_on_valid = 0
        
        self.state = reset_board()
        if random.random() < self.probability_of_regular_spawn: spawn_balls(self.state, override=random.randint(SPAWN_COUNT, GRID_SIZE**2))
        else: spawn_balls(self.state, override=random.randint(self.custom_spawn_range[0], self.custom_spawn_range[1]))
        
        return self.get_observation(), {}

    def get_observation(self):
        cnn_obs = np.reshape(self.state["board"], (GRID_SIZE, GRID_SIZE, 1)).astype(np.uint8)
        vector_obs = self.state["next_colors"].astype(np.uint8)
        
        return {
            "cnn_features": cnn_obs,
            "vector_features": vector_obs,
            "action_mask": self.action_masks()
        }

    def action_masks(self):
        flat_board = self.state['board'].flatten()
        start_mask = flat_board != 0
        end_mask = flat_board == 0
        
        valid_pairs_mask = np.outer(start_mask, end_mask)

        if not valid_pairs_mask.any():
            valid_pairs_mask[0, 0] = True

        return valid_pairs_mask.flatten()

    def step(self, action):
        start_square, end_square = divmod(int(action), GRID_SIZE**2)
        sr, sc = divmod(start_square, GRID_SIZE)
        er, ec = divmod(end_square, GRID_SIZE)

        move_result = apply_move(self.state, (sr, sc), (er, ec))
        validity = move_result["validity"]
        num_empty = len(self.state["empties"])
        self.num_attempted_moves += 1

        if validity == -1:
            base_reward = self.clear_ball_reward
            # Add bonus for longer lines
            cleared_count = len(move_result["cleared"])
            if cleared_count == 4: base_reward += 75
            elif cleared_count >= 5: base_reward += 475
            reward = base_reward
            self.num_valid_moves += 1
        elif validity == 0:
            reward = self.valid_move_reward
            self.num_valid_moves += 1
            self.num_balls_on_valid = GRID_SIZE**2 - len(self.state["empties"])
        elif validity == 0.5:
            reward = self.invalid_move_reward
        else:
            reward = self.invalid_move_reward
        
        last_start, last_end = divmod(self.last_action, GRID_SIZE**2)
        is_repeat = (start_square == last_end) and (end_square == last_start)
        if is_repeat:
            reward += self.repeat_move_reward
            self.num_repeated_moves += 1
        self.last_action = action
        
        if self.scale_rewards: reward = reward * self.num_valid_moves

        # --- Definitive Fix: Clip the reward to a stable range ---
        final_reward = np.clip(reward, -30, 30)

        terminated = (
            (num_empty == 0) or
            (num_empty == GRID_SIZE**2) or
            (num_empty <= GRID_SIZE**2 * (1 - self.end_game_board_percentage)) or
            (self.num_attempted_moves >= self.end_game_num_attempted_moves) or
            (self.num_valid_moves >= self.end_game_num_valid_moves) or
            (self.num_repeated_moves >= self.end_game_num_repeated_moves)
        )
        truncated = False

        observation = self.get_observation()
        info = {
            "validity": validity,
            "reward": final_reward, # Return the clipped reward
            "score": self.state["score"],
            "terminated": terminated,
            "truncated": truncated,
            "num_balls_on_valid": self.num_balls_on_valid,
            "is_repeat": is_repeat,
            "num_cleared": len(move_result["cleared"])
        }
        return observation, final_reward, terminated, truncated, info

    def render(self, mode="human"): pass