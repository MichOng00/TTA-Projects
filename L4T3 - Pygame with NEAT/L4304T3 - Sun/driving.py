import pygame
import math
import sys
import neat

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

GRASS_COLOR = pygame.Color(2, 105, 31, 255)

pygame.init()

class Car(pygame.sprite.Sprite): # subclass of pygame Sprite class
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        # self.drive_state = 0
        self.drive_state = True
        self.vel_vector = pygame.math.Vector2(0.8, 0) # start driving to the right 
        self.angle = 0
        self.rotation_vel = 5 # turning speed
        self.direction = 0
        self.time_since_death = 0

    def update(self):
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        

    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * 6
        # if self.drive_state == 1:
        #     self.rect.center += self.vel_vector * 6
        # if self.drive_state == -1:
        #     self.rect.center -= self.vel_vector * 6
    def rotate(self):
        if self.direction == 1: # turn right
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1: # turn left
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center = self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x, y = int(self.rect.centerx), int(self.rect.centery)
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
        pygame.draw.circle(SCREEN, GRASS_COLOR, (x, y), 3)

    def check_collision(self):
        x, y = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x, y))
            if color_at_car == GRASS_COLOR:
                print("You touched grass! Respawning...")
                self.respawn()

    def respawn(self):
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0) 
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0

car = pygame.sprite.GroupSingle(Car())

def eval_genomes():
    start_time = pygame.time.get_ticks()
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0,0))

        user_input = pygame.key.get_pressed()
        if sum(user_input) <= 1: # no keys are pressed
            car.sprite.drive_state = False # stop driving
            car.sprite.direction = 0 # reset direction to straight
        # if user_input[pygame.K_UP]:
        #     car.sprite.drive_state = 1
        # if user_input[pygame.K_DOWN]:
        #     car.sprite.drive_state = -1
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1

        car.update()
        car.sprite.check_collision()
        car.draw(SCREEN)
        pygame.display.update()

eval_genomes()

# pastebin.com/pV6VWrMf