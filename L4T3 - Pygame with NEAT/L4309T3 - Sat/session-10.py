import pygame
import math
import sys
import neat
import random

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track_andy.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

GRASS_COLOR = pygame.Color(2, 105, 31, 255)

LEAVE_RADIUS = 150
FINISH_RADIUS = 40
MIN_LAP_STEPS = 100
LAP_FITNESS_BASE = 1000
DEATH_PENALTY = 20

MIN_SPEED = 3
MAX_SPEED = 15

pygame.init()

font = pygame.font.Font(None, 36)

class Car(pygame.sprite.Sprite):
    def __init__(self, color):
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
        self.alive = True
        self.color = color
        self.speed = MIN_SPEED
        self.start_pos = START_POS
        # lap tracking
        self.distance_traveled = 0
        self.spawn_step = 0
        self.left_start = False
        self.lap_complete = False
        self.lap_time = None
        self.laps_completed = 0

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
            # self.rect.center += self.vel_vector * (6 + self.gear)
            movement = self.vel_vector * self.speed
            self.rect.center += movement
            self.distance_traveled += movement.length()

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

    def get_radar_distances(self):
        distances = []
        for radar_angle in (-60, -30, 0, 30, 60):
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
            distances.append(length / 200)
        return distances

    def apply_controls(self, throttle, steering):
        self.drive_state = True
        throttle_norm = max(0, min(1, throttle))
        self.speed = MIN_SPEED + throttle_norm * (MAX_SPEED - MIN_SPEED)
        if steering > 0.2:
            self.direction = 1
        elif steering < -0.2:
            self.direction = -1
        else:
            self.direction = 0

    def distance_from_start(self):
        dx = self.rect.center[0] - START_POS[0]
        dy = self.rect.center[1] - START_POS[1]
        return math.hypot(dx, dy)

    def check_lap(self, current_step):
        dist = self.distance_from_start()
        # check for starting lap
        if not self.left_start:
            if dist > LEAVE_RADIUS:
                self.left_start = True
        # check for finishing lap
        elif not self.lap_complete and dist < FINISH_RADIUS:
            lap_time = current_step - self.spawn_step
            if lap_time > MIN_LAP_STEPS:
                self.lap_complete = True
                self.lap_time = lap_time

    def reset_lap(self, current_step):
        self.laps_completed += 1
        self.lap_complete = False
        self.left_start = False
        self.spawn_step = current_step
        
def main(genomes, config):
    global cars, ge, nets
    cars = []
    ge = []
    nets = []
    clock = pygame.time.Clock()

    for id, genome in genomes:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        car = Car(color)
        cars.append(car)
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    steps = 0
    max_steps = 5000
    fitness_threshold = 5000

    while run and len(cars) > 0 and steps < max_steps:
        steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0,0))

        for i, car in enumerate(cars):
            if car.alive:
                # ge[i].fitness += 1
                # if ge[i].fitness >= fitness_threshold:
                #     car.alive = False

                inputs = car.get_radar_distances()
                output = nets[i].activate(inputs)
                throttle = output[0]
                steering = output[1]        # number from -1 to 1
                car.apply_controls(throttle, steering)
                car.update()
                car.check_lap(steps)

                progress_fitness = car.distance_traveled / 100
                if progress_fitness > ge[i].fitness:
                    ge[i].fitness = progress_fitness

                finished_lap_this_frame = car.lap_complete
                if finished_lap_this_frame:
                    lap_fitness = LAP_FITNESS_BASE - car.lap_time
                    ge[i].fitness += lap_fitness
                    car.reset_lap(steps)

                SCREEN.blit(car.image, car.rect)
                pygame.draw.circle(SCREEN, car.color, car.rect.center, 5)

                if not car.alive:
                    if not finished_lap_this_frame:
                        ge[i].fitness = max(0, ge[i].fitness - DEATH_PENALTY)
                    cars.pop(i)
                    ge.pop(i)
                    nets.pop(i)
                    break

        text = font.render(f"Cars alive: {len(cars)}", True, (0,0,0))
        SCREEN.blit(text, (20,20))
        if len(cars) > 0:
            fit_text = font.render(f"Fitness: {ge[0].fitness:.2f}", True, (0,0,0))
            SCREEN.blit(fit_text, (20,80))

        pygame.display.update()
        clock.tick(60)

def run_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.run(main, 80)

run_neat("config_car.txt") # tinyurl.com/56jks6m6