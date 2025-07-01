import pygame, sys, time, random, numpy as np
from collections import deque


# --- Constants ---

GRID_SIZE = 9
SPAWN_COUNT = 3
LINE_LENGTH = 2

CELL_SIZE = 60
MARGIN = 2
HEADER_FONT_SIZE = 28
HEADER_FONT_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (80, 80, 80)
ANIM_DELAY = 0.08
BALL_RENDER_COLORS = [
    None,
    (255, 59, 48),    # red
    (255, 204, 0),    # yellow
    (76, 217, 100),   # green
    (90, 200, 250),   # light blue
    (0, 122, 255),    # blue
    (120, 86, 214),   # purple
    (153, 102, 51)    # brown
]

grid_pixel = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * MARGIN
HEADER_PREVIEW_HEIGHT = CELL_SIZE + 2 * MARGIN
HEADER_HEIGHT = HEADER_PREVIEW_HEIGHT + 2 * MARGIN
SCREEN_WIDTH = grid_pixel
SCREEN_HEIGHT = HEADER_HEIGHT + grid_pixel


# --- Game mechanics ---

def reset_board():
    return {
        'board': np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8), # 0 is empty, 1 through NUM_COLORS is the color
        'empties': {(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)},
        'score': 0,
        'moves': 0,
        'next_colors': np.random.randint(1, len(BALL_RENDER_COLORS), size=SPAWN_COUNT)
    }

def empty_cells(state): return state['empties']

def spawn_balls(state, override=False): # Move balls in state['next_colors'] into random empty cells and generates new next_colors
    if override:
        n_spawn = min(len(state['empties']), override)
        if n_spawn == 0: return []
        spawn_positions = random.sample(list(state['empties']), n_spawn)
        for (r, c) in spawn_positions:
            color = random.randint(1, len(BALL_RENDER_COLORS) - 1)
            state['board'][r, c] = color
            state['empties'].discard((r, c))
        return spawn_positions
    else:
        n_spawn = min(len(state['empties']), SPAWN_COUNT)
        if n_spawn == 0: return []
        spawn_positions = random.sample(list(state['empties']), n_spawn)
        for i, (r, c) in enumerate(spawn_positions):
            state['board'][r, c] = state['next_colors'][i]
            state['empties'].discard((r, c))
        state['next_colors'] = np.random.randint(1, len(BALL_RENDER_COLORS), size=SPAWN_COUNT)
        return spawn_positions

def is_valid_move(board, start, end): # Returns 0 if start is occupied and end is empty (0.5 if no path exists); 1 if start and end are occupied; returns 2 if start and end are empty; returns 3 if start is empty and end is occupied
    sr, sc = start; er, ec = end
    if board[sr][sc] != 0 and board[er][ec] != 0: return 1
    elif board[sr][sc] == 0 and board[er][ec] == 0: return 2
    elif board[sr][sc] == 0 and board[er][ec] != 0: return 3
    visited = [[False]*GRID_SIZE for _ in range(GRID_SIZE)]
    dq = deque([start]); visited[sr][sc] = True
    while dq:
        r, c = dq.popleft()
        if (r,c) == (er,ec): return 0
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc = r+dr, c+dc
            if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE and not visited[nr][nc] and board[nr][nc]==0:
                visited[nr][nc] = True
                dq.append((nr,nc))
    return 0.5

def find_matches(board, positions): # Check for lines going through the given positions and returns a list of cell tuples to clear
    to_clear = set()
    directions = [((1,0),(-1,0)), ((0,1),(0,-1)), ((1,1),(-1,-1)), ((1,-1),(-1,1))]
    
    for (r0,c0) in positions:
        color = board[r0, c0]
        if color == 0: continue
        for dir1, dir2 in directions:
            line = [(r0, c0)]
            for dr, dc in (dir1, dir2):
                r, c = r0+dr, c0+dc
                while 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r, c] == color:
                    line.append((r, c))
                    r += dr; c += dc
            if len(line) >= LINE_LENGTH: to_clear.update(line)
    
    return list(to_clear)

def apply_move(state, start, end): # Returns valididty of move; if valid, moves ball, increments moves, clears matches, updates score, and spawns new balls
    board = state['board']
    validity = is_valid_move(board, start, end)
    if validity != 0: return validity
    
    board[end], board[start] = board[start], 0
    state['empties'].discard(end); state['empties'].add(start)
    state['moves'] += 1
    
    cleared = find_matches(board, [end])
    state['score'] += len(cleared)
    if len(cleared) > 0:
        board[tuple(zip(*cleared))] = 0
        for cell in cleared: state['empties'].add(cell)
        return -1
    
    cleared_due_to_spawned_balls = find_matches(board, spawn_balls(state))
    if len(cleared_due_to_spawned_balls) > 0: board[tuple(zip(*cleared_due_to_spawned_balls))] = 0
    return 0


# --- Rendering --- 

def find_path(board, start, end):
    sr, sc = start; er, ec = end
    if board[er, ec] != 0: return None
    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    parents = {start: None}
    queue = deque([start]); visited[sr][sc] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == (er, ec):
            path = []; node = (er, ec)
            while node:
                path.append(node)
                node = parents[node]
            return path[::-1]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not visited[nr][nc] and board[nr, nc] == 0:
                visited[nr][nc] = True
                parents[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    
    return None

def draw_state(screen, font, state, selected=None, highlight=None, anim_board=None, ai_move=None, ai_start_color=None, ai_end_color=None):
    board = anim_board if anim_board is not None else state['board']
    next_colors = state['next_colors']; score = state['score']; moves = state['moves']
    screen.fill((30, 30, 30))
    # Title bar
    x = MARGIN; y_preview = MARGIN
    for color in next_colors:
        rect = pygame.Rect(x, y_preview, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (60, 60, 60), rect, 1)
        if color:
            pygame.draw.circle(screen, BALL_RENDER_COLORS[color],
                               (x + CELL_SIZE // 2, y_preview + CELL_SIZE // 2), CELL_SIZE // 2 - 4)
        x += CELL_SIZE + MARGIN
    occupied = GRID_SIZE * GRID_SIZE - len(state['empties'])
    header = f"Score: {score}   Move: {moves}   Occ: {occupied}/{GRID_SIZE*GRID_SIZE}"
    txt = font.render(header, True, HEADER_FONT_COLOR)
    screen.blit(txt, (x + MARGIN, y_preview + (CELL_SIZE - txt.get_height()) // 2))
    # Grid and balls
    yoff = HEADER_HEIGHT
    for i in range(GRID_SIZE + 1):
        xi = MARGIN + i * (CELL_SIZE + MARGIN)
        pygame.draw.line(screen, (60, 60, 60), (xi, yoff + MARGIN), (xi, yoff + grid_pixel - MARGIN))
        pygame.draw.line(screen, (60, 60, 60), (MARGIN, yoff + xi), (grid_pixel - MARGIN, yoff + xi))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            v = board[r, c]
            if v:
                pygame.draw.circle(screen, BALL_RENDER_COLORS[v],
                                   (MARGIN + c * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
                                    yoff + MARGIN + r * (CELL_SIZE + MARGIN) + CELL_SIZE // 2),
                                   CELL_SIZE // 2 - 4)
    # Highlight AI move
    if ai_move:
        (sr, sc), (er, ec) = ai_move
        # AI start
        rect_start = pygame.Rect(
            MARGIN + sc * (CELL_SIZE + MARGIN),
            yoff + MARGIN + sr * (CELL_SIZE + MARGIN),
            CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, ai_start_color, rect_start, 3)
        # AI end
        rect_end = pygame.Rect(
            MARGIN + ec * (CELL_SIZE + MARGIN),
            yoff + MARGIN + er * (CELL_SIZE + MARGIN),
            CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, ai_end_color, rect_end, 3)

    # Highlight selection (drawn on top of AI move if they overlap)
    if selected:
        sr, sc = selected
        rect = pygame.Rect(
            MARGIN + sc * (CELL_SIZE + MARGIN),
            yoff + MARGIN + sr * (CELL_SIZE + MARGIN),
            CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, HEADER_FONT_COLOR, rect, 3)
    # Highlight destination (animation)
    if highlight:
        hr, hc = highlight
        rect = pygame.Rect(
            MARGIN + hc * (CELL_SIZE + MARGIN),
            yoff + MARGIN + hr * (CELL_SIZE + MARGIN),
            CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)
    pygame.display.flip()

def animate_move(screen, font, state, start, end):
    path = find_path(state['board'], start, end)
    if not path: return
    base = state['board'].copy(); color = base[start]
    for step in path:
        temp = base.copy()
        temp[start] = 0; temp[step] = color
        draw_state(screen, font, state, selected=start, highlight=end, anim_board=temp)
        time.sleep(ANIM_DELAY)