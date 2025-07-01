import math, random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
from strike5_engine import reset_board, empty_cells, is_valid_move, apply_move, spawn_balls, SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_FONT_SIZE, HEADER_HEIGHT, MARGIN, CELL_SIZE, GRID_SIZE, SPAWN_COUNT

class Strike5Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, clear_ball_reward, repeat_move_reward, occupied_empty_reward, occupied_empty_no_path_reward, occupied_occupied_reward, empty_empty_reward, empty_occupied_reward, end_game_board_percentage=2, end_game_num_valid_moves=math.inf, end_game_num_invalid_moves=math.inf, end_game_num_repeated_moves=math.inf, end_game_num_attempted_moves=math.inf, custom_spawn_range=(3, 3), probability_of_regular_spawn=0, scale_rewards=False):
        super().__init__()
        self.clear_ball_reward = clear_ball_reward
        self.repeat_move_reward = repeat_move_reward
        self.occupied_empty_reward = occupied_empty_reward
        self.occupied_empty_no_path_reward = occupied_empty_no_path_reward
        self.occupied_occupied_reward = occupied_occupied_reward
        self.empty_empty_reward = empty_empty_reward
        self.empty_occupied_reward = empty_occupied_reward
        self.end_game_board_percentage = end_game_board_percentage
        self.end_game_num_valid_moves = end_game_num_valid_moves
        self.end_game_num_invalid_moves = end_game_num_invalid_moves
        self.end_game_num_repeated_moves = end_game_num_repeated_moves
        self.end_game_num_attempted_moves = end_game_num_attempted_moves
        self.custom_spawn_range = custom_spawn_range
        self.probability_of_regular_spawn = probability_of_regular_spawn
        self.scale_rewards = scale_rewards

        self.observation_space = spaces.Box(low=0, high=7, shape=(GRID_SIZE**2 + SPAWN_COUNT,), dtype=np.int8,)
        self.action_space = MultiDiscrete([GRID_SIZE**2, GRID_SIZE**2])
        self.state = None
        self.num_attempted_moves = 0
        self.num_valid_moves = 0
        self.num_invalid_moves = 0
        self.num_balls_on_valid = 0
        self.last_move = [0, 0]
        self.num_repeated_moves = 0

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.num_attempted_moves = self.num_valid_moves = self.num_invalid_moves = self.num_repeated_moves = self.num_balls_on_valid = 0
        self.state = reset_board()
        if random.random() < self.probability_of_regular_spawn: spawn_balls(self.state, override=random.randint(SPAWN_COUNT, GRID_SIZE**2))
        else: spawn_balls(self.state, override=random.randint(self.custom_spawn_range[0], self.custom_spawn_range[1]))
        return self.get_observation(), {}

    def get_observation(self): return np.concatenate([self.state["board"].flatten(), self.state["next_colors"]]).astype(np.int8)

    def step(self, action):
        sr, sc = divmod(int(action[0]), GRID_SIZE)
        er, ec = divmod(int(action[1]), GRID_SIZE)

        validity = apply_move(self.state, (sr, sc), (er, ec))
        num_empty = len(self.state["empties"])
        self.num_attempted_moves += 1

        if validity == -1: reward = self.clear_ball_reward; self.num_valid_moves += 1
        elif validity == 0: reward = self.occupied_empty_reward; self.num_valid_moves += 1; self.num_balls_on_valid = GRID_SIZE**2 - len(self.state["empties"])
        elif validity == 0.5: reward = self.occupied_empty_no_path_reward; self.num_invalid_moves += 1
        elif validity == 1: reward = self.occupied_occupied_reward; self.num_invalid_moves += 1
        elif validity == 2: reward = self.empty_empty_reward; self.num_invalid_moves += 1
        else: reward = self.empty_occupied_reward; self.num_invalid_moves += 1

        if (action[0] == self.last_move[0] and action[1] == self.last_move[1]) or (action[0] == self.last_move[1] and action[1] == self.last_move[0]): reward += self.repeat_move_reward; self.num_repeated_moves += 1
        self.last_move = [action[0], action[1]]
        
        if self.scale_rewards: reward = reward * self.num_valid_moves
        truncated = False
        terminated = (num_empty == 0) or (num_empty <= GRID_SIZE**2 * (1 - self.end_game_board_percentage)) or self.num_attempted_moves >= self.end_game_num_attempted_moves or self.num_valid_moves >= self.end_game_num_valid_moves or self.num_invalid_moves >= self.end_game_num_invalid_moves or self.num_repeated_moves >= self.end_game_num_repeated_moves
        observation = self.get_observation()
        info = {
            "validity": validity,
            "reward": reward,
            "score": self.state["score"],
            "terminated": terminated,
            "truncated": truncated,
            "num_balls_on_valid": self.num_balls_on_valid,
            "is_repeat": (action[0] == self.last_move[0] and action[1] == self.last_move[1]) or (action[0] == self.last_move[1] and action[1] == self.last_move[0])
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"): pass