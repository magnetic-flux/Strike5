import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.moves = 0
        self.clears = 0
        self.validity_0s = 0
        self.validity_05s = 0
        self.num_repeat_moves = 0

        self.current_game_reward = 0
        self.current_game_valid_moves = 0
        
        self.game_rewards = []
        self.game_lengths_valid_moves = []


    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.moves += 1
        self.current_game_reward += info["reward"]

        validity = info["validity"]
        if validity == -1:
            self.clears += 1
            self.current_game_valid_moves += 1
        elif validity == 0:
            self.validity_0s += 1
            self.current_game_valid_moves += 1
        elif validity == 0.5:
            self.validity_05s += 1
        
        if info["is_repeat"]:
            self.num_repeat_moves += 1

        if info["truncated"] or info["terminated"]:
            self.game_rewards.append(self.current_game_reward)
            self.game_lengths_valid_moves.append(self.current_game_valid_moves)
            self.current_game_reward = 0
            self.current_game_valid_moves = 0

        return True
    
    def _on_rollout_end(self):
        total_games = len(self.game_lengths_valid_moves)
        if total_games == 0:
            print("No completed games this rollout")
            total_games = 1
            self.game_lengths_valid_moves.append(self.current_game_valid_moves)

        self.logger.record("Telemetry/1. Average game length", np.mean(self.game_lengths_valid_moves))
        self.logger.record("Telemetry/2. Average clears per game", self.clears / total_games)
        self.logger.record("Telemetry/3. Average reward per game", np.mean(self.game_rewards))
        self.logger.record("Telemetry/4. Fraction of moves that are valid with no clear", self.validity_0s / self.moves)
        self.logger.record("Telemetry/5. Fraction moves that clear balls", self.clears / self.moves)
        self.logger.record("Telemetry/6. Fraction of moves that are invalid with no path", self.validity_05s / self.moves)
        self.logger.record("Telemetry/7. Average number of repeat moves per game", self.num_repeat_moves / total_games)

        self.moves = 0
        self.clears = 0
        self.validity_0s = 0
        self.validity_05s = 0
        self.num_repeat_moves = 0
        self.current_game_reward = 0
        self.current_game_valid_moves = 0
        self.game_rewards = []
        self.game_lengths_valid_moves = []