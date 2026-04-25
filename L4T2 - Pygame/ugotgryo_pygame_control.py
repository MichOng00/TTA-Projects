import pygame
from ugot import ugot

pygame.init()

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

got = ugot.UGOT()
got.initialize("192.168.1.200")

x, y = WIDTH // 2, HEIGHT // 2
speed = 5

def clamp(v, low, high):
    return max(low, min(high, v))

def read_gyro():
    data = got.read_gyro_data()
    pitch = data[0]
    roll = data[1]
    yaw = data[2]
    return pitch, roll, yaw

center_pitch, center_roll, center_yaw = read_gyro()

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                center_pitch, center_roll, center_yaw = read_gyro()

    keys = pygame.key.get_pressed()

    pitch, roll, yaw = read_gyro()

    # Roll controls left/right
    move_x = (roll - center_roll) / 20

    # Pitch controls up/down
    move_y = (pitch - center_pitch) / 20

    # Keyboard backup
    if keys[pygame.K_LEFT]:
        move_x = -1
    if keys[pygame.K_RIGHT]:
        move_x = 1
    if keys[pygame.K_UP]:
        move_y = -1
    if keys[pygame.K_DOWN]:
        move_y = 1

    x += move_x * speed
    y += move_y * speed

    x = clamp(x, 20, WIDTH - 20)
    y = clamp(y, 20, HEIGHT - 20)

    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (50, 120, 255), (int(x), int(y)), 20)

    pygame.display.flip()

pygame.quit()