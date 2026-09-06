"""NEAT self-driving car simulation.

Usage:
    python car_neat.py train                 # train from scratch
    python car_neat.py train --resume ckpt/neat-checkpoint-15
    python car_neat.py watch                 # watch the saved winner drive
"""

import argparse
import json
import math
import os
import pickle
import random
import sys

import neat
import pygame

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None  # sphere-traced radar falls back to per-pixel walking

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735

ASSETS_DIR = "Assets/not_car"
TRACK_PATH = os.path.join(ASSETS_DIR, "track_shortcut.png")
CAR_PATH = os.path.join(ASSETS_DIR, "car.png")
WAYPOINTS_PATH = os.path.join(ASSETS_DIR, "waypoints.json")

START_POS = (350, 600)
GRASS_COLOR = pygame.Color(2, 105, 31, 255)

LEAVE_RADIUS = 150
FINISH_RADIUS = 40
MIN_LAP_STEPS = 100
LAP_FITNESS_BASE = 1000
DEATH_PENALTY = 20
WAYPOINT_RADIUS = 60
WAYPOINT_FITNESS = 50  # awarded once per waypoint reached, in sequence

MIN_SPEED = 3
MAX_SPEED = 15

RADAR_ANGLES = (-60, -30, 0, 30, 60)
RADAR_MAX_LENGTH = 200
MAX_STEPS_PER_GEN = 5000

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
WINNER_FILE = "winner.pkl"


# --------------------------------------------------------------------------
# Track / collision setup
# --------------------------------------------------------------------------
def load_image(path, label):
    try:
        return pygame.image.load(path)
    except pygame.error as e:
        print(f"Could not load {label} at '{path}': {e}")
        sys.exit(1)


def load_waypoints(path):
    """Optional list of [x, y] points along the track centerline, in
    order. If present, fitness is based on how far along this sequence
    a car gets (robust to a car just farming distance by circling in
    place). If absent, we fall back to a cruder distance-traveled
    metric and print a note."""
    if os.path.exists(path):
        with open(path) as f:
            points = json.load(f)
        return [tuple(p) for p in points]
    return None


class Track:
    """Owns the track image, the static grass lookup, and (if scipy is
    available) a distance-to-grass field used to speed up radar casts."""

    def __init__(self, track_path, screen_size):
        self.width, self.height = screen_size
        self.surface = pygame.transform.scale(
            load_image(track_path, "track image"), screen_size
        )
        self.grass_rgb = (GRASS_COLOR.r, GRASS_COLOR.g, GRASS_COLOR.b)

        # Snapshot the *static* track once. Checking collisions/radar
        # against this instead of the live screen avoids a subtle bug:
        # the live screen also has cars/radar-lines drawn on it each
        # frame, so a car could "see" another car's pixels instead of
        # grass. It's also far faster than repeated surface.get_at()
        # calls, which lock the surface every time.
        self.array = pygame.surfarray.array3d(self.surface)  # (W, H, 3)

        self.distance_field = None
        if ndimage is not None and np is not None:
            drivable = ~np.all(self.array == self.grass_rgb, axis=-1)
            # For each drivable pixel, distance (in px) to the nearest
            # grass pixel. Lets radar rays take steps of that size
            # instead of walking one pixel at a time (classic "sphere
            # tracing" against a distance field).
            self.distance_field = ndimage.distance_transform_edt(drivable)

    def is_grass(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            r, g, b = self.array[x, y]
            return (r, g, b) == self.grass_rgb
        return True  # off-screen counts as a wall

    def free_distance(self, x, y):
        """Distance to nearest grass from (x, y), or 0 if unknown/on grass."""
        if self.distance_field is not None and 0 <= x < self.width and 0 <= y < self.height:
            return self.distance_field[x, y]
        return 0


# --------------------------------------------------------------------------
# Car
# --------------------------------------------------------------------------
class Car(pygame.sprite.Sprite):
    def __init__(self, track, car_image, color, waypoints=None):
        super().__init__()
        self.track = track
        self.original_image = car_image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0  # 0: straight, 1/-1: left/right
        self.alive = True
        self.color = color
        self.speed = MIN_SPEED

        # lap tracking
        self.distance_traveled = 0
        self.spawn_step = 0
        self.left_start = False
        self.lap_complete = False
        self.lap_time = None
        self.laps_completed = 0

        # waypoint-based progress (preferred fitness signal, if available)
        self.waypoints = waypoints
        self.next_waypoint = 0

        # cached each frame so a ray is only cast once, not twice
        self.radar_distances = [1.0] * len(RADAR_ANGLES)
        self.radar_endpoints = [self.rect.center] * len(RADAR_ANGLES)
        self._collision_points = (self.rect.center, self.rect.center)

    def update(self):
        self.drive()
        self.rotate()
        self.cast_radar()
        self.check_wall_collision()
        self.check_waypoints()

    def drive(self):
        if self.drive_state:
            movement = self.vel_vector * self.speed
            self.rect.center += movement
            self.distance_traveled += movement.length()

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def cast_radar(self):
        cx, cy = self.rect.center
        for idx, radar_angle in enumerate(RADAR_ANGLES):
            length = 0.0
            x, y = float(cx), float(cy)
            heading = math.radians(self.angle + radar_angle)
            dx, dy = math.cos(heading), -math.sin(heading)

            while length < RADAR_MAX_LENGTH:
                step = self.track.free_distance(int(x), int(y))
                if step < 1:
                    # Either off the distance field (fallback mode) or
                    # right at a grass boundary: fall back to a single
                    # pixel step so we still terminate correctly.
                    step = 1
                    if self.track.is_grass(int(x), int(y)):
                        break
                length = min(length + step, RADAR_MAX_LENGTH)
                x = cx + dx * length
                y = cy - (-dy) * length  # dy already includes the sign flip above

            self.radar_distances[idx] = length / RADAR_MAX_LENGTH
            self.radar_endpoints[idx] = (int(x), int(y))

    def check_wall_collision(self):
        length = 30
        cx, cy = self.rect.center
        right = (
            int(cx + math.cos(math.radians(self.angle + 18)) * length),
            int(cy - math.sin(math.radians(self.angle + 18)) * length),
        )
        left = (
            int(cx + math.cos(math.radians(self.angle - 18)) * length),
            int(cy - math.sin(math.radians(self.angle - 18)) * length),
        )
        if self.track.is_grass(*right) or self.track.is_grass(*left):
            self.alive = False
        self._collision_points = (left, right)

    def check_waypoints(self):
        if not self.waypoints or self.next_waypoint >= len(self.waypoints):
            return
        target = self.waypoints[self.next_waypoint]
        if math.hypot(self.rect.centerx - target[0], self.rect.centery - target[1]) < WAYPOINT_RADIUS:
            self.next_waypoint += 1
            if self.next_waypoint >= len(self.waypoints):
                self.next_waypoint = 0  # loop back around for lap 2, 3, ...

    def progress_fitness(self):
        """Fitness contribution from forward progress. Prefers the
        waypoint sequence (robust to a car exploiting fitness by
        circling in a small loop, since only reaching the *next*
        waypoint counts); falls back to raw distance traveled if no
        waypoints file was provided."""
        if self.waypoints:
            # whole waypoints passed, plus partial credit toward the next one
            target = self.waypoints[self.next_waypoint]
            dist_to_next = math.hypot(self.rect.centerx - target[0], self.rect.centery - target[1])
            partial = max(0.0, 1.0 - dist_to_next / 400)
            return self.laps_completed * len(self.waypoints) * WAYPOINT_FITNESS \
                + self.next_waypoint * WAYPOINT_FITNESS + partial * WAYPOINT_FITNESS
        return self.distance_traveled / 100

    def draw(self, surface, draw_radar=True):
        surface.blit(self.image, self.rect)
        pygame.draw.circle(surface, self.color, self.rect.center, 5)
        if draw_radar:
            for endpoint in self.radar_endpoints:
                pygame.draw.line(surface, (255, 255, 255), self.rect.center, endpoint, 1)
                pygame.draw.circle(surface, (0, 255, 0), endpoint, 3)
            for p in self._collision_points:
                pygame.draw.circle(surface, (0, 255, 255), p, 4)

    def get_radar_distances(self):
        return self.radar_distances

    def apply_controls(self, throttle, steering):
        self.drive_state = True
        throttle_norm = max(0.0, min(1.0, throttle))
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
        if not self.left_start:
            if dist > LEAVE_RADIUS:
                self.left_start = True
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


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------
def draw_hud(surface, font, cars_alive, best_fitness, generation, turbo):
    lines = [
        f"Cars alive: {cars_alive}",
        f"Best fitness: {best_fitness:.2f}",
        f"Generation: {generation}",
    ]
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, (0, 0, 0)), (20, 20 + i * 34))
    if turbo:
        surface.blit(font.render("TURBO (T to toggle)", True, (200, 0, 0)), (20, 20 + len(lines) * 34))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
class Trainer:
    def __init__(self, screen, track, car_image, waypoints, checkpoint_every):
        self.screen = screen
        self.track = track
        self.car_image = car_image
        self.waypoints = waypoints
        self.font = pygame.font.Font(None, 36)
        self.generation = 0
        self.turbo = False
        self.checkpoint_every = checkpoint_every

    def eval_genomes(self, genomes, config):
        self.generation += 1
        clock = pygame.time.Clock()

        cars, ge, nets = [], [], []
        for _id, genome in genomes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cars.append(Car(self.track, self.car_image, color, self.waypoints))
            ge.append(genome)
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
            genome.fitness = 0

        steps = 0
        while cars and steps < MAX_STEPS_PER_GEN:
            steps += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    self.turbo = not self.turbo

            dead_indices = []
            for i, car in enumerate(cars):
                output = nets[i].activate(car.get_radar_distances())
                car.apply_controls(output[0], output[1])
                car.update()
                car.check_lap(steps)

                progress = car.progress_fitness()
                if progress > ge[i].fitness:
                    ge[i].fitness = progress

                if car.lap_complete:
                    ge[i].fitness += LAP_FITNESS_BASE - car.lap_time
                    car.reset_lap(steps)
                elif not car.alive:
                    ge[i].fitness = max(0, ge[i].fitness - DEATH_PENALTY)

                if not car.alive:
                    dead_indices.append(i)

            # Remove dead cars after the loop (in reverse), instead of
            # mutating the list mid-iteration -- popping+breaking during
            # the loop meant only one death was handled per frame and
            # every car after it silently skipped that frame's update.
            for i in reversed(dead_indices):
                cars.pop(i)
                ge.pop(i)
                nets.pop(i)

            if not self.turbo:
                self.screen.blit(self.track.surface, (0, 0))
                for car in cars:
                    car.draw(self.screen)
                best_fitness = max((g.fitness for g in ge), default=0)
                draw_hud(self.screen, self.font, len(cars), best_fitness, self.generation, self.turbo)
                pygame.display.update()
                clock.tick(60)


def run_training(args, screen, track, car_image, waypoints):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    if args.resume:
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    else:
        pop = neat.Population(config)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(args.checkpoint_every, filename_prefix=CHECKPOINT_PREFIX))

    trainer = Trainer(screen, track, car_image, waypoints, args.checkpoint_every)
    winner = pop.run(trainer.eval_genomes, args.generations)

    with open(args.winner_file, "wb") as f:
        pickle.dump((winner, config), f)
    print(f"\nSaved best genome to {args.winner_file}")
    print(winner)


# --------------------------------------------------------------------------
# Watch mode: replay a saved winner, no training
# --------------------------------------------------------------------------
def run_watch(args, screen, track, car_image, waypoints):
    if not os.path.exists(args.winner_file):
        print(f"No saved genome found at '{args.winner_file}'. Train first.")
        sys.exit(1)

    with open(args.winner_file, "rb") as f:
        genome, config = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    car = Car(track, car_image, (255, 60, 60), waypoints)
    steps = 0

    while car.alive and steps < MAX_STEPS_PER_GEN:
        steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        output = net.activate(car.get_radar_distances())
        car.apply_controls(output[0], output[1])
        car.update()
        car.check_lap(steps)
        if car.lap_complete:
            print(f"Lap {car.laps_completed + 1} complete in {car.lap_time} steps")
            car.reset_lap(steps)

        screen.blit(track.surface, (0, 0))
        car.draw(screen)
        screen.blit(font.render(f"Laps: {car.laps_completed}", True, (0, 0, 0)), (20, 20))
        pygame.display.update()
        clock.tick(60)

    print(f"Car crashed after {steps} steps, {car.laps_completed} laps completed.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "watch"], nargs="?", default="train")
    p.add_argument("--config", default="config_car.txt", help="NEAT config file (tinyurl.com/56jks6m6)")
    p.add_argument("--track", default=TRACK_PATH)
    p.add_argument("--car-image", default=CAR_PATH)
    p.add_argument("--waypoints", default=WAYPOINTS_PATH)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", help="Path to a checkpoint file to resume training from")
    p.add_argument("--winner-file", default=WINNER_FILE)
    return p.parse_args()


def main():
    args = parse_args()
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Self-Driving Car")

    track = Track(args.track, (SCREEN_WIDTH, SCREEN_HEIGHT))
    car_image = load_image(args.car_image, "car image")
    waypoints = load_waypoints(args.waypoints)
    if waypoints is None:
        print(
            f"No waypoints file at '{args.waypoints}' -- falling back to raw "
            "distance-traveled fitness. Note this can be gamed by a car "
            "circling in a small loop; add a waypoints.json "
            "(list of [x, y] points along the track) for a more robust signal."
        )
    if track.distance_field is None:
        print("scipy not available -- radar will walk pixel-by-pixel (slower). "
              "Install scipy for faster sphere-traced radar casts.")

    if args.mode == "train":
        run_training(args, screen, track, car_image, waypoints)
    else:
        run_watch(args, screen, track, car_image, waypoints)


if __name__ == "__main__":
    main()