import pygame
import math
import sys
import neat

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600) # starting position of the car

GRASS_COLOR = (2, 105, 31)

pygame.init()

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0) # moving to the right at speed 0.8
        self.angle = 0 # where car is facing at the moment
        self.rotation_vel = 5 # how fast car rotates
        self.direction = 0 # whether you are turning left or right or going straight
        self.time_since_death = 0
        self.alive = True

    def update(self):
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def collision(self):
        length = 30
        right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length)
        left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length)

        right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)

        right = (right_x, right_y)
        left = (left_x, left_y)

        if 0 <= right_x < SCREEN_WIDTH and 0 <= right_y < SCREEN_HEIGHT:
            if SCREEN.get_at(right) == GRASS_COLOR:
                self.alive = False
                print("dead")

        if 0 <= left_x < SCREEN_WIDTH and 0 <= left_y < SCREEN_HEIGHT:
            if SCREEN.get_at(left) == GRASS_COLOR:
                self.alive = False
                print("dead")

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), left, 4)


    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * 6

    def rotate(self):
        # check if turning left or right
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        # rotate image and update position
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center = self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while (
            0 <= x < SCREEN_WIDTH and
            0 <= y < SCREEN_HEIGHT and
            not SCREEN.get_at((x, y)) == GRASS_COLOR and
            length < 200
        ):
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0), (x, y), 3)

    def check_collision(self):
        x, y = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x,y)) # color under car centre
            if color_at_car == GRASS_COLOR: # you die
                print("You touched grass! Respawning...")
                self.respawn()

    def respawn(self):
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.angle = 0
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.drive_state = False

    def draw_info(self, start_time):
        font = pygame.font.Font(None, 36)

        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        death_time = (pygame.time.get_ticks() - self.time_since_death) // 1000
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        death_minutes = death_time // 60
        death_seconds = death_time % 60
        timer_text = f"Time: {minutes:02}:{seconds:02}"
        lap_text = f"Lap Time: {death_minutes:02}:{death_seconds:02}"

        # draw background box
        info_box = pygame.Rect(SCREEN_WIDTH - 220, 10, 210, 70)
        pygame.draw.rect(SCREEN, (0,0,0), info_box, border_radius=5)
        pygame.draw.rect(SCREEN, (255,255,255), info_box, 2, border_radius=5)

        # blit text
        timer_surface = font.render(timer_text, True, (255, 255, 255))
        lap_surface = font.render(lap_text, True, (255, 255, 255))
        SCREEN.blit(timer_surface, (SCREEN_WIDTH - 210, 20))
        SCREEN.blit(lap_surface, (SCREEN_WIDTH - 210, 40))

car = pygame.sprite.GroupSingle(Car())

def eval_genomes():
    start_time = pygame.time.get_ticks()
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        user_input = pygame.key.get_pressed()
        if sum(user_input) <= 1:
            car.sprite.drive_state = False
            car.sprite.direction = 0

        # drive
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True

        # steering
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1

        car.update()
        car.sprite.check_collision()
        car.draw(SCREEN) # comes from pygame.sprite.Sprite
        car.sprite.draw_info(start_time)
        pygame.display.update()

eval_genomes()