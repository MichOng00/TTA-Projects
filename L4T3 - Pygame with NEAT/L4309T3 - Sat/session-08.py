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

font = pygame.font.Font(None, 36)

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0 # 0: straight, 1/-1: left/right
        self.gear = 1
        self.time_since_death = 0

    def update(self):
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def collision(self):
        length = 30
        coll_right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle+18))*length)
        coll_left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle-18))*length)
        coll_right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle+18))*length)
        coll_left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle-18))*length)

        right = (coll_right_x, coll_right_y)
        left = (coll_left_x, coll_left_y)

        if SCREEN.get_at(right) == GRASS_COLOR or SCREEN.get_at(left) == GRASS_COLOR:
            self.alive = False

        pygame.draw.circle(SCREEN, (0,255,255,0), right, 4)
        pygame.draw.circle(SCREEN, (0,255,255,0), left, 4)

    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * (6 + self.gear)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
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

        pygame.draw.line(SCREEN, (255,255,255), self.rect.center, (x,y), 1)
        pygame.draw.circle(SCREEN, (0,255,0,0), (x,y), 3)

    def check_collision(self):
        x, y  = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x+5, y+5)) # shift to let the car die 
            if color_at_car == GRASS_COLOR:
                print("touch grass")
                self.respawn()

    def respawn(self):
        global time_since_death
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.angle = 0
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.drive_state = False

    def draw_info(self, start_time):
        global time_since_death

        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        death_time = (pygame.time.get_ticks() - self.time_since_death) // 1000
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        death_minutes = death_time // 60
        death_seconds = death_time % 60

        timer_text = f"Time: {minutes:02}:{seconds:02}"
        lap_time_text = f"Lap time: {death_minutes:02}:{death_seconds:02}"

        info_box = pygame.Rect(SCREEN_WIDTH-220, 10, 210, 70)
        pygame.draw.rect(SCREEN, (0,0,0), info_box, border_radius=5)
        pygame.draw.rect(SCREEN, (255,255,255), info_box, 2, border_radius=5)

        timer_surface = font.render(timer_text, True, (255,255,255))
        lap_surface = font.render(lap_time_text, True, (255,255,255))
        SCREEN.blit(timer_surface, (SCREEN_WIDTH-210, 20))
        SCREEN.blit(lap_surface, (SCREEN_WIDTH-210, 40))
        

car = pygame.sprite.GroupSingle(Car())

def main():
    start_time = pygame.time.get_ticks()
    shifted = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    car.sprite.gear += 1
                    car.sprite.gear = min(car.sprite.gear, 5)
                if event.key == pygame.K_LCTRL:
                    car.sprite.gear -= 1
                    car.sprite.gear = max(car.sprite.gear, 1)

        SCREEN.blit(TRACK, (0,0))

        user_input = pygame.key.get_pressed()
        if sum(user_input) <= 1:
            car.sprite.drive_state = False
            car.sprite.direction = 0
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1

        car.update()
        car.sprite.check_collision()
        car.draw(SCREEN)
        car.sprite.draw_info(start_time)

        gear_text = f"Gear: {car.sprite.gear}"
        gear_surface = font.render(gear_text, True, (255,255,255))
        SCREEN.blit(gear_surface, (20, 20))

        pygame.display.update()

    pygame.quit()
    sys.exit()
main()

# pastebin.com/DuQ1DWTt