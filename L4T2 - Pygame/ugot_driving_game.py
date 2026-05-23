import random
import pygame

# Optional gyro support from the ugot module.
try:
    from ugot import ugot
    got = ugot.UGOT()
    got.initialize("192.168.1.181")
    GYRO_AVAILABLE = True
except:
    got = None
    GYRO_AVAILABLE = False

pygame.init()

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UGOT Driving Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 32)
large_font = pygame.font.SysFont(None, 56)

ROAD_WIDTH = 320
ROAD_LEFT = (WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH
LANE_MARK_WIDTH = 8
LANE_MARK_HEIGHT = 40
LANE_MARK_SPACING = 30

CAR_WIDTH = 40
CAR_HEIGHT = 60
CAR_Y = HEIGHT - CAR_HEIGHT - 20
CAR_COLOR = (200, 30, 30)

OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 40
OBSTACLE_COLOR = (30, 30, 180)
OBSTACLE_SPEED = 5
SPAWN_INTERVAL = 900

MAX_X = ROAD_RIGHT - CAR_WIDTH - 20
MIN_X = ROAD_LEFT + 20


def clamp(value, low, high):
    return max(low, min(high, value))


def read_gyro():
    if not GYRO_AVAILABLE:
        return 0.0, 0.0, 0.0
    data = got.read_gyro_data()
    if data is None or len(data) < 3:
        return 0.0, 0.0, 0.0
    return data[0], data[1], data[2]


def draw_road(scroll_offset):
    screen.fill((80, 170, 80))
    pygame.draw.rect(screen, (40, 40, 40), (ROAD_LEFT, 0, ROAD_WIDTH, HEIGHT))
    pygame.draw.rect(screen, (255, 255, 255), (ROAD_LEFT, 0, 10, HEIGHT))
    pygame.draw.rect(screen, (255, 255, 255), (ROAD_RIGHT - 10, 0, 10, HEIGHT))

    for y in range(-LANE_MARK_HEIGHT, HEIGHT, LANE_MARK_HEIGHT + LANE_MARK_SPACING):
        draw_y = y + scroll_offset % (LANE_MARK_HEIGHT + LANE_MARK_SPACING)
        pygame.draw.rect(screen, (250, 250, 0), (WIDTH // 2 - LANE_MARK_WIDTH // 2, draw_y, LANE_MARK_WIDTH, LANE_MARK_HEIGHT))


def draw_car(x):
    rect = pygame.Rect(int(x), CAR_Y, CAR_WIDTH, CAR_HEIGHT)
    pygame.draw.rect(screen, CAR_COLOR, rect)
    pygame.draw.polygon(screen, (255, 255, 255), [(rect.centerx - 10, rect.top + 10), (rect.centerx + 10, rect.top + 10), (rect.centerx, rect.top)])


def draw_obstacle(obstacle):
    pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle)
    pygame.draw.rect(screen, (255, 255, 255), obstacle, 2)


def show_text(message, y, color=(255, 255, 255), font_obj=None):
    if font_obj is None:
        font_obj = font
    text_surface = font_obj.render(message, True, color)
    screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, y))


def main():
    center_pitch, center_roll, center_yaw = read_gyro()
    x = WIDTH // 2 - CAR_WIDTH // 2
    speed = 6
    scroll_offset = 0
    score = 0
    game_over = False
    last_spawn = pygame.time.get_ticks()
    obstacles = []

    running = True
    while running:
        dt = clock.tick(60)
        scroll_offset += speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    center_pitch, center_roll, center_yaw = read_gyro()
                    if game_over:
                        game_over = False
                        x = WIDTH // 2 - CAR_WIDTH // 2
                        score = 0
                        obstacles.clear()
                        last_spawn = pygame.time.get_ticks()

        keys = pygame.key.get_pressed()
        pitch, roll, yaw = read_gyro()

        # Use gyroscope roll for steering
        move_x = (roll - center_roll) / 15.0
        if keys[pygame.K_LEFT]:
            move_x = -1.8
        if keys[pygame.K_RIGHT]:
            move_x = 1.8

        if not game_over:
            x += move_x * speed
            x = clamp(x, MIN_X, MAX_X)
            score += dt * 0.01

            now = pygame.time.get_ticks()
            if now - last_spawn > SPAWN_INTERVAL:
                last_spawn = now
                lane_x = random.choice([ROAD_LEFT + 40, WIDTH // 2 - OBSTACLE_WIDTH // 2, ROAD_RIGHT - 40 - OBSTACLE_WIDTH])
                obstacles.append(pygame.Rect(lane_x, -OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

            for obstacle in obstacles:
                obstacle.y += OBSTACLE_SPEED

            obstacles = [obs for obs in obstacles if obs.y < HEIGHT + OBSTACLE_HEIGHT]

            player_rect = pygame.Rect(int(x), CAR_Y, CAR_WIDTH, CAR_HEIGHT)
            if any(player_rect.colliderect(obs) for obs in obstacles):
                game_over = True

        draw_road(scroll_offset)
        draw_car(x)
        for obstacle in obstacles:
            draw_obstacle(obstacle)

        show_text(f"Score: {int(score)}", 10)
        show_text("Press R to center gyro / restart", HEIGHT - 30, (240, 240, 240))

        if not GYRO_AVAILABLE:
            show_text("Gyro unavailable: use LEFT/RIGHT arrows", 40, (255, 220, 0))

        if game_over:
            show_text("GAME OVER", HEIGHT // 2 - 40, (255, 80, 80), large_font)
            show_text("Press R to restart", HEIGHT // 2 + 20, (255, 255, 255))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
