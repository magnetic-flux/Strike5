import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # General counters, reset each rollout
        self.moves = 0
        self.clears = 0
        self.validity_0s = 0
        self.validity_05s = 0
        self.num_repeat_moves = 0
        self.total_balls_cleared_on_clear_moves = 0

        # Per-game trackers
        self.current_game_reward = 0
        self.current_game_valid_moves = 0
        
        # Rollout accumulators
        self.game_rewards = []
        self.game_lengths_valid_moves = []


    def _on_step(self) -> bool:
        # We assume a single environment for detailed info logging
        info = self.locals["infos"][0]
        
        self.moves += 1
        self.current_game_reward += info["reward"]

        validity = info["validity"]
        if validity == -1: # Move resulted in a clear
            self.clears += 1
            self.current_game_valid_moves += 1
            self.total_balls_cleared_on_clear_moves += info["num_cleared"]
        elif validity == 0: # Valid move, no clear
            self.validity_0s += 1
            self.current_game_valid_moves += 1
        elif validity == 0.5: # Invalid move, no path
            self.validity_05s += 1
        
        if info["is_repeat"]:
            self.num_repeat_moves += 1

        # When a game ends, log its stats and reset per-game trackers
        if info["truncated"] or info["terminated"]:
            self.game_rewards.append(self.current_game_reward)
            self.game_lengths_valid_moves.append(self.current_game_valid_moves)
            self.current_game_reward = 0
            self.current_game_valid_moves = 0

        return True
    
    def _on_rollout_end(self):
        total_games = len(self.game_lengths_valid_moves)
        # Handle case where no games finished in the rollout
        if total_games == 0:
            print("No completed games this rollout")
            total_games = 1 # Avoid division by zero, use stats from the single ongoing game
            self.game_lengths_valid_moves.append(self.current_game_valid_moves)
            self.game_rewards.append(self.current_game_reward)

        # Calculate metrics for the completed rollout
        avg_balls_cleared = (
            self.total_balls_cleared_on_clear_moves / self.clears
            if self.clears > 0
            else 0
        )

        # Log metrics to TensorBoard
        self.logger.record("Telemetry/1. Average game length", np.mean(self.game_lengths_valid_moves))
        self.logger.record("Telemetry/2. Average clears per game", self.clears / total_games)
        self.logger.record("Telemetry/3. Average reward per game", np.mean(self.game_rewards))
        self.logger.record("Telemetry/4. Fraction of moves that are valid with no clear", self.validity_0s / self.moves if self.moves > 0 else 0)
        self.logger.record("Telemetry/5. Fraction moves that clear balls", self.clears / self.moves if self.moves > 0 else 0)
        self.logger.record("Telemetry/6. Fraction of moves that are invalid with no path", self.validity_05s / self.moves if self.moves > 0 else 0)
        self.logger.record("Telemetry/7. Average balls cleared on clear", avg_balls_cleared)
        self.logger.record("Telemetry/8. Average number of repeat moves per game", self.num_repeat_moves / total_games)

        # Reset all rollout counters
        self.moves = 0
        self.clears = 0
        self.validity_0s = 0
        self.validity_05s = 0
        self.num_repeat_moves = 0
        self.total_balls_cleared_on_clear_moves = 0
        self.current_game_reward = 0
        self.current_game_valid_moves = 0
        self.game_rewards = []
        self.game_lengths_valid_moves = []
