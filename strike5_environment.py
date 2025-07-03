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

        self.observation_space = Dict({
            "action_mask": Box(low=0, high=1, shape=(GRID_SIZE**4,), dtype=bool),
            "observation": Box(low=0, high=7, shape=(GRID_SIZE**2 + SPAWN_COUNT,), dtype=np.int8)
        })
        self.action_space = Discrete(GRID_SIZE**4)
        
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

    def get_observation(self): return {"observation": np.concatenate([self.state["board"].flatten(), self.state["next_colors"]]).astype(np.int8), "action_mask": self.action_masks()}

    def action_masks(self):
        flat_board = self.state['board'].flatten()
        start_mask = flat_board != 0  # Can only start from an occupied square
        end_mask = flat_board == 0    # Can only end on an empty square
        
        mask = np.outer(start_mask, end_mask).flatten()

        # Failsafe to prevent an all-False mask
        if not np.any(mask): mask[0] = True
            
        return mask

    def step(self, action):
        start_square, end_square = divmod(action, GRID_SIZE**2)
        sr, sc = divmod(start_square, GRID_SIZE)
        er, ec = divmod(end_square, GRID_SIZE)

        move_result = apply_move(self.state, (sr, sc), (er, ec))
        validity = move_result["validity"]
        num_empty = len(self.state["empties"])
        self.num_attempted_moves += 1

        if validity == -1:
            reward = self.clear_ball_reward
            self.num_valid_moves += 1
        elif validity == 0:
            reward = self.valid_move_reward
            self.num_valid_moves += 1
            self.num_balls_on_valid = GRID_SIZE**2 - len(self.state["empties"])
        elif validity == 0.5:
            reward = self.invalid_move_reward
        else:
            print("this should never execute")
            reward = self.invalid_move_reward
        
        last_start, last_end = divmod(self.last_action, GRID_SIZE**2)
        is_repeat = start_square == last_end and end_square == last_start
        if is_repeat:
            reward += self.repeat_move_reward
            self.num_repeated_moves += 1
        self.last_action = action

        #TODO: aaaa
        if len(move_result["cleared"]) == 4: reward = 100
        elif len(move_result["cleared"]) == 5: reward = 500
        
        if self.scale_rewards: reward = reward * self.num_valid_moves

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
            "reward": reward,
            "score": self.state["score"],
            "terminated": terminated,
            "truncated": truncated,
            "num_balls_on_valid": self.num_balls_on_valid,
            "is_repeat": is_repeat,
            "num_cleared": len(move_result["cleared"])
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"): pass