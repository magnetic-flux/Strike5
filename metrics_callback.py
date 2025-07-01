from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.moves = 0
        self.validity_0s = self.validity_05s = self.validity_1s = self.validity_2s = self.validity_3s = 0
        self.reward_sum = 0
        self.clears = 0
        self.current_game_length = 0
        self.num_repeat_moves = 0
        self.game_lengths = []
        self.num_balls_on_valid = []
        self.num_balls_on_clear = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.moves += 1; self.current_game_length += 1
        
        if info["truncated"] or info["terminated"]:
            self.game_lengths.append(self.current_game_length)
            self.current_game_length = 0

        if info["validity"] == -1: self.clears += 1; self.num_balls_on_valid.append(info["num_balls_on_valid"]); self.num_balls_on_clear.append(info["num_balls_on_valid"]) 
        if info["validity"] == 0: self.validity_0s += 1; self.num_balls_on_valid.append(info["num_balls_on_valid"])
        elif info["validity"] == 0.5: self.validity_05s += 1
        elif info["validity"] == 1: self.validity_1s += 1
        elif info["validity"] == 2: self.validity_2s += 1
        else: self.validity_3s += 1

        if info["is_repeat"]: self.num_repeat_moves += 1

        self.reward_sum += info["reward"]
        return True
    
    def _on_rollout_end(self):
        self.game_lengths.append(self.current_game_length)
        print(self.num_balls_on_valid)
        print(self.num_balls_on_clear)
        self.logger.record("Telemetry/1. Number of clears per game", self.clears / len(self.game_lengths))
        self.logger.record("Telemetry/2. OCCUPIED to EMPTY (0)", self.validity_0s / self.moves)
        self.logger.record("Telemetry/3. OCCUPIED to EMPTY but no path (0.5)", self.validity_05s / self.moves)
        self.logger.record("Telemetry/4. OCCUPIED to OCCUPIED (1)", self.validity_1s / self.moves)
        self.logger.record("Telemetry/5. EMPTY to EMPTY (2)", self.validity_2s / self.moves)
        self.logger.record("Telemetry/6. EMPTY to OCCUPIED (3)", self.validity_3s / self.moves)
        self.logger.record("Telemetry/7. Average game length", sum(self.game_lengths) / len(self.game_lengths))
        self.logger.record("Telemetry/8. Number of repeat moves", self.num_repeat_moves)
        self.logger.record("Telemetry/9. Number of valid moves per game", (self.validity_0s / len(self.game_lengths)) if self.game_lengths else 0)
        
        self.moves = 0
        self.validity_0s = 0
        self.validity_05s = 0
        self.validity_1s = 0
        self.validity_2s = 0
        self.validity_3s = 0
        self.reward_sum = 0
        self.clears = 0
        self.done_count = 0
        self.current_game_length = 0
        self.num_repeat_moves = 0
        self.game_lengths = []
        self.num_balls_on_valid = []
        self.num_balls_on_clear = []