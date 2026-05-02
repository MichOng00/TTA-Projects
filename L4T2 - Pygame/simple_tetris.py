import pygame
import random

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 800
GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 40

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple Tetris")
clock = pygame.time.Clock()

COLORS = [
    (0, 255, 255),  # Cyan
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Green
    (128, 0, 128),  # Purple
    (255, 0, 0),    # Red
]

SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
]


def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]


def create_piece():
    shape = random.choice(SHAPES)
    color = random.choice(COLORS)
    return {"shape": shape, "color": color, "x": GRID_WIDTH // 2 - len(shape[0]) // 2, "y": 0}


def valid_position(grid, piece, offset_x=0, offset_y=0):
    for row_index, row in enumerate(piece["shape"]):
        for col_index, cell in enumerate(row):
            if cell:
                x = piece["x"] + col_index + offset_x
                y = piece["y"] + row_index + offset_y
                if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                    return False
                if grid[y][x] != (0, 0, 0):
                    return False
    return True


def lock_piece(grid, piece):
    for row_index, row in enumerate(piece["shape"]):
        for col_index, cell in enumerate(row):
            if cell:
                x = piece["x"] + col_index
                y = piece["y"] + row_index
                if 0 <= y < GRID_HEIGHT:
                    grid[y][x] = piece["color"]


def clear_lines(grid):
    new_grid = [row for row in grid if any(cell == (0, 0, 0) for cell in row)]
    lines_cleared = GRID_HEIGHT - len(new_grid)
    for _ in range(lines_cleared):
        new_grid.insert(0, [(0, 0, 0) for _ in range(GRID_WIDTH)])
    return new_grid, lines_cleared


def draw_grid(surface, grid, piece):
    surface.fill((30, 30, 30))
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            color = grid[y][x]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, (50, 50, 50), rect, 1)
            if color != (0, 0, 0):
                pygame.draw.rect(surface, color, rect.inflate(-4, -4))

    for row_index, row in enumerate(piece["shape"]):
        for col_index, cell in enumerate(row):
            if cell:
                x = piece["x"] + col_index
                y = piece["y"] + row_index
                if y >= 0:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(surface, piece["color"], rect.inflate(-4, -4))


def draw_hud(surface, score, level, lines):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    level_text = font.render(f"Level: {level}", True, (255, 255, 255))
    lines_text = font.render(f"Lines: {lines}", True, (255, 255, 255))
    surface.blit(score_text, (10, 10))
    surface.blit(level_text, (10, 50))
    surface.blit(lines_text, (10, 90))


def main():
    grid = [[(0, 0, 0) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    current_piece = create_piece()
    next_piece = create_piece()
    score = 0
    lines_cleared = 0
    level = 1
    fall_time = 0
    fall_speed = 500
    running = True
    game_over = False

    while running:
        dt = clock.tick(60)
        fall_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and not game_over:
                if event.key == pygame.K_LEFT:
                    if valid_position(grid, current_piece, offset_x=-1):
                        current_piece["x"] -= 1
                elif event.key == pygame.K_RIGHT:
                    if valid_position(grid, current_piece, offset_x=1):
                        current_piece["x"] += 1
                elif event.key == pygame.K_DOWN:
                    if valid_position(grid, current_piece, offset_y=1):
                        current_piece["y"] += 1
                elif event.key == pygame.K_UP:
                    rotated_shape = rotate(current_piece["shape"])
                    old_shape = current_piece["shape"]
                    current_piece["shape"] = rotated_shape
                    if not valid_position(grid, current_piece):
                        current_piece["shape"] = old_shape
                elif event.key == pygame.K_SPACE:
                    while valid_position(grid, current_piece, offset_y=1):
                        current_piece["y"] += 1
                    fall_time = fall_speed

        if not game_over and fall_time >= fall_speed:
            fall_time = 0
            if valid_position(grid, current_piece, offset_y=1):
                current_piece["y"] += 1
            else:
                lock_piece(grid, current_piece)
                grid, cleared = clear_lines(grid)
                if cleared > 0:
                    lines_cleared += cleared
                    score += cleared * 100
                    level = 1 + lines_cleared // 10
                    fall_speed = max(100, 500 - (level - 1) * 40)
                current_piece = next_piece
                next_piece = create_piece()
                if not valid_position(grid, current_piece):
                    game_over = True

        draw_grid(screen, grid, current_piece)
        draw_hud(screen, score, level, lines_cleared)

        if game_over:
            font = pygame.font.Font(None, 72)
            text = font.render("Game Over", True, (255, 255, 255))
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(text, rect)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
