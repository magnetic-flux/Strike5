import pygame, sys, time, random, numpy as np
from sb3_contrib import MaskablePPO
from strike5_engine import reset_board, empty_cells, is_valid_move, apply_move, spawn_balls, draw_state, animate_move, SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_FONT_SIZE, HEADER_HEIGHT, MARGIN, CELL_SIZE, GRID_SIZE, SPAWN_COUNT

AI_HELP = True
MODEL_PATH = "./logs_sb3/strike5_ppo_1000000_steps.zip"
AI_START_COLOR = (50, 255, 50)
AI_END_COLOR = (255, 165, 0)

model = None
if AI_HELP:
    try:
        model = MaskablePPO.load(MODEL_PATH)
        print("AI Helper model loaded successfully from:", MODEL_PATH)
    except Exception as e:
        print("Error loading AI model:", e)
        AI_HELP = False

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Strike5_Manual_Play')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, HEADER_FONT_SIZE)

state = reset_board()
spawn_balls(state)
selected = None; running = True
ai_move = None

while running:
    moved = False
    
    if AI_HELP and model:
        # 1. Create the observation in the correct dictionary format with a batch dimension of 1
        obs_board = np.reshape(state["board"], (1, GRID_SIZE, GRID_SIZE, 1)).astype(np.uint8)
        obs_vector = np.reshape(state["next_colors"], (1, SPAWN_COUNT)).astype(np.uint8)
        observation = {
            "cnn_features": obs_board,
            "vector_features": obs_vector
        }
        
        # 2. Manually create the action mask
        flat_board = state['board'].flatten()
        start_mask = flat_board != 0
        end_mask = flat_board == 0
        
        # Handle edge case where no moves are possible
        if not np.any(start_mask) or not np.any(end_mask):
            start_mask = np.zeros(GRID_SIZE**2, dtype=bool)
            end_mask = np.zeros(GRID_SIZE**2, dtype=bool)
            start_mask[0] = True # Make a dummy valid move
            end_mask[0] = True
        
        # Combine masks and add a batch dimension
        action_masks = np.concatenate([start_mask, end_mask])
        action_masks = np.expand_dims(action_masks, axis=0)

        # 3. Predict using both the observation AND the action mask
        actions, _ = model.predict(observation, action_masks=action_masks, deterministic=True)

        # 4. Un-batch the result: predict returns a batch of actions, so we take the first one
        action = actions[0]

        sr, sc = divmod(int(action[0]), GRID_SIZE)
        er, ec = divmod(int(action[1]), GRID_SIZE)
        ai_move = ((sr, sc), (er, ec))

    for evt in pygame.event.get():
        if evt.type == pygame.QUIT: running = False
        elif evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
            x, y = evt.pos
            if y > HEADER_HEIGHT:
                c = (x - MARGIN) // (CELL_SIZE + MARGIN)
                r = (y - HEADER_HEIGHT - MARGIN) // (CELL_SIZE + MARGIN)
                
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    if state['board'][r, c] != 0:
                        selected = (r, c)
                        draw_state(screen, font, state, selected=selected, ai_move=ai_move, ai_start_color=AI_START_COLOR, ai_end_color=AI_END_COLOR)
                    elif selected and is_valid_move(state['board'], selected, (r, c)) == 0:
                        animate_move(screen, font, state, selected, (r, c))
                        apply_move(state, selected, (r, c))
                        selected = None; moved = True

    if not moved: draw_state(screen, font, state, selected=selected, ai_move=ai_move, ai_start_color=AI_START_COLOR, ai_end_color=AI_END_COLOR)
    
    clock.tick(60)

pygame.quit(); sys.exit()