import pygame
import math
import sys
import neat
import pickle
import random

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

GRASS_COLOR = pygame.Color(2, 105, 31, 255)

# --- Lap-based fitness tuning ----------------------------------------
# These control how "fastest lap" fitness works. A car keeps driving
# after finishing a lap and can attempt more laps for the rest of the
# generation; its fitness is always its single fastest completed lap.
# Tune these to your track.
LEAVE_RADIUS = 150         # px a car must get from START_POS before a lap can start counting
FINISH_RADIUS = 40         # px from START_POS that counts as crossing the finish line
MIN_LAP_STEPS = 100        # frames that must pass before a lap can register (avoids instant re-trigger)
LAP_FITNESS_BASE = 100000  # baseline so ANY finished lap beats ANY unfinished attempt
DEATH_PENALTY = 20         # fitness knocked off a car's score when it crashes into the grass

pygame.init()

font = pygame.font.Font(None, 36)

class Car(pygame.sprite.Sprite):
    def __init__(self, color=None):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center=START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.gear = 1
        self.time_since_death = 0
        self.alive = True
        self.color = color if color is not None else (255, 255, 255)
        self.start_pos = START_POS          # store start for distance check

        # --- lap tracking state ---
        self.distance_traveled = 0.0        # cumulative px moved this attempt
        self.spawn_step = 0                 # frame count when the current lap attempt began
        self.left_start = False             # has it gotten far enough from start yet?
        self.lap_complete = False           # has it just finished a lap this frame?
        self.lap_time = None                # frames taken to complete the most recent lap
        self.laps_completed = 0             # total laps finished this generation

    def update(self):
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def collision(self):
        length = 30
        coll_right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length)
        coll_left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length)
        coll_right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        coll_left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)

        right = (coll_right_x, coll_right_y)
        left = (coll_left_x, coll_left_y)

        if SCREEN.get_at(right) == GRASS_COLOR or SCREEN.get_at(left) == GRASS_COLOR:
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), left, 4)

    def drive(self):
        if self.drive_state:
            movement = self.vel_vector * (6 + self.gear)
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
        self.rect = self.image.get_rect(center=self.rect.center)

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
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

    def check_collision(self):
        x, y = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x + 5, y + 5))
            if color_at_car == GRASS_COLOR:
                print("touch grass")
                self.respawn()

    def respawn(self):
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.angle = 0
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.drive_state = False

    def draw_info(self, start_time):
        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        death_time = (pygame.time.get_ticks() - self.time_since_death) // 1000
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        death_minutes = death_time // 60
        death_seconds = death_time % 60

        timer_text = f"Time: {minutes:02}:{seconds:02}"
        lap_time_text = f"Lap time: {death_minutes:02}:{death_seconds:02}"

        info_box = pygame.Rect(SCREEN_WIDTH - 220, 10, 210, 70)
        pygame.draw.rect(SCREEN, (0, 0, 0), info_box, border_radius=5)
        pygame.draw.rect(SCREEN, (255, 255, 255), info_box, 2, border_radius=5)

        timer_surface = font.render(timer_text, True, (255, 255, 255))
        lap_surface = font.render(lap_time_text, True, (255, 255, 255))
        SCREEN.blit(timer_surface, (SCREEN_WIDTH - 210, 20))
        SCREEN.blit(lap_surface, (SCREEN_WIDTH - 210, 40))

    # NEAT helpers
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
            distances.append(length / 200.0)
        return distances

    def apply_controls(self, throttle, steering):
        self.drive_state = True      # always move forward
        if steering > 0.2:
            self.direction = 1
        elif steering < -0.2:
            self.direction = -1
        else:
            self.direction = 0

    # helper to check distance from start
    def distance_from_start(self):
        dx = self.rect.center[0] - self.start_pos[0]
        dy = self.rect.center[1] - self.start_pos[1]
        return math.hypot(dx, dy)

    def check_lap(self, current_step):
        """Track whether the car has left the start area and, once it
        has, whether it has now come back around to cross the finish
        line near START_POS. Sets self.lap_complete / self.lap_time
        the moment a lap is finished."""
        dist = self.distance_from_start()
        if not self.left_start:
            if dist > LEAVE_RADIUS:
                self.left_start = True
        elif not self.lap_complete and dist < FINISH_RADIUS:
            lap_time = current_step - self.spawn_step
            if lap_time > MIN_LAP_STEPS:
                self.lap_complete = True
                self.lap_time = lap_time

    def reset_lap(self, current_step):
        """Start timing a fresh lap after finishing one, without killing
        the car — it keeps driving and can attempt another (hopefully
        faster) lap for the rest of the generation."""
        self.laps_completed += 1
        self.lap_complete = False
        self.left_start = False
        self.spawn_step = current_step

# ----------------------------------------------------------------------
# NEAT evaluation
# ----------------------------------------------------------------------
def eval_genomes(genomes, config):
    global cars, ge, nets
    cars = []
    ge = []
    nets = []
    clock = pygame.time.Clock()

    for genome_id, genome in genomes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        car = Car(color)
        cars.append(car)
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    steps = 0
    max_steps = 5000                   # maximum frames per generation

    while run and len(cars) > 0 and steps < max_steps:
        steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        for i, car in enumerate(cars):
            if car.alive:
                inputs = car.get_radar_distances()
                output = nets[i].activate(inputs)
                throttle = output[0]
                steering = output[1] if len(output) > 1 else 0.0
                car.apply_controls(throttle, steering)
                car.update()
                car.check_lap(steps)

                # Fitness is driven by "fastest lap", not by how long the
                # car survives. Before a lap is finished we give a small
                # distance-based score purely so NEAT still has a
                # gradient to select on (otherwise every unfinished
                # genome ties at 0 and evolution can't tell them apart).
                # We only ever raise fitness (never lower it), so a car
                # keeps credit for its best result even after it moves
                # on to try another lap.
                progress_fitness = car.distance_traveled / 100.0
                if progress_fitness > ge[i].fitness:
                    ge[i].fitness = progress_fitness

                finished_lap_this_frame = car.lap_complete
                if finished_lap_this_frame:
                    # LAP_FITNESS_BASE is large enough that ANY finished
                    # lap beats ANY unfinished attempt, and subtracting
                    # lap_time means a faster lap always scores higher
                    # than a slower one. Because we only raise fitness,
                    # the car's score always reflects its single fastest
                    # lap of the generation, however many it attempts.
                    lap_fitness = LAP_FITNESS_BASE - car.lap_time
                    if lap_fitness > ge[i].fitness:
                        ge[i].fitness = lap_fitness
                    car.reset_lap(steps)   # keep driving, try another lap

                # Draw the car (blit image and a colored dot)
                SCREEN.blit(car.image, car.rect)
                pygame.draw.circle(SCREEN, car.color, car.rect.center, 5)

                if not car.alive:
                    if not finished_lap_this_frame:
                        # Crashing into the grass still costs fitness,
                        # instead of the car just keeping whatever
                        # partial-progress score it had built up.
                        ge[i].fitness = max(0.0, ge[i].fitness - DEATH_PENALTY)
                    cars.pop(i)
                    ge.pop(i)
                    nets.pop(i)
                    break  # break for loop because list changed

        # Statistics
        text = font.render(f"Cars alive: {len(cars)}", True, (0, 0, 0))
        SCREEN.blit(text, (20, 20))
        if len(cars) > 0:
            fit_text = font.render(f"Fitness: {ge[0].fitness:.1f}", True, (0, 0, 0))
            SCREEN.blit(fit_text, (20, 50))
            step_text = font.render(f"Step: {steps}", True, (0, 0, 0))
            SCREEN.blit(step_text, (20, 80))
            lap_text = font.render(f"Laps: {cars[0].laps_completed}", True, (0, 0, 0))
            SCREEN.blit(lap_text, (20, 110))

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
    pop.run(eval_genomes, 30)

# ----------------------------------------------------------------------
# Original keyboard-controlled version (commented out)
# ----------------------------------------------------------------------
# car = pygame.sprite.GroupSingle(Car())
#
# def main():
#     start_time = pygame.time.get_ticks()
#     shifted = False
#     run = True
#     while run:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_LSHIFT:
#                     car.sprite.gear += 1
#                     car.sprite.gear = min(car.sprite.gear, 5)
#                 if event.key == pygame.K_LCTRL:
#                     car.sprite.gear -= 1
#                     car.sprite.gear = max(car.sprite.gear, 1)
#         SCREEN.blit(TRACK, (0,0))
#         user_input = pygame.key.get_pressed()
#         if sum(user_input) <= 1:
#             car.sprite.drive_state = False
#             car.sprite.direction = 0
#         if user_input[pygame.K_UP]:
#             car.sprite.drive_state = True
#         if user_input[pygame.K_RIGHT]:
#             car.sprite.direction = 1
#         if user_input[pygame.K_LEFT]:
#             car.sprite.direction = -1
#         car.update()
#         car.sprite.check_collision()
#         car.draw(SCREEN)
#         car.sprite.draw_info(start_time)
#         gear_text = f"Gear: {car.sprite.gear}"
#         gear_surface = font.render(gear_text, True, (255,255,255))
#         SCREEN.blit(gear_surface, (20, 20))
#         pygame.display.update()
#     pygame.quit()
#     sys.exit()
# main()

# ----------------------------------------------------------------------
# Start NEAT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_neat("config-car.txt")

# pastebin.com/DuQ1DWTt
# customise track