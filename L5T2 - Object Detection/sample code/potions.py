"""
Potion Mixing Game — Pygame + MediaPipe HandLandmarker (Vision Tasks API)
Controls: pinch index finger + thumb to grab potions, bring two together to mix.
"""

import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import math
import random
import sys
import time
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Download hand_landmarker.task model if not present ────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print(f"Downloading hand_landmarker model → {MODEL_PATH} …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Pygame init ────────────────────────────────────────────────────────────────
pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("✨ Potion Mixer")
clock = pygame.time.Clock()

# ── Fonts ──────────────────────────────────────────────────────────────────────
font_lg   = pygame.font.SysFont("Arial", 32, bold=True)
font_md   = pygame.font.SysFont("Arial", 22)
font_sm   = pygame.font.SysFont("Arial", 16)
font_xl   = pygame.font.SysFont("Arial", 48, bold=True)
font_emoji= pygame.font.SysFont("Segoe UI Emoji", 36)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = (18, 15, 35)
PANEL    = (30, 24, 55)
WHITE    = (255, 255, 255)
GREY     = (140, 130, 170)

POTION_DEFS = {
    # name         : (r, g, b,   emoji, display_name)
    "red"          : ((220,  60,  60), "🔴", "Red"),
    "blue"         : (( 60,  60, 220), "🔵", "Blue"),
    "yellow"       : ((220, 200,  40), "🟡", "Yellow"),
    "green"        : (( 40, 180,  80), "🟢", "Green"),
    "purple"       : ((160,  40, 220), "🟣", "Purple"),
    "orange"       : ((230, 130,  30), "🟠", "Orange"),
    "white"        : ((230, 230, 230), "⚪", "White"),
    "cyan"         : (( 40, 200, 210), "🩵", "Cyan"),
    "pink"         : ((230,  80, 160), "🩷", "Pink"),
    "brown"        : ((130,  80,  40), "🟤", "Brown"),
    "gold"         : ((220, 185,  20), "✨", "Gold"),
    "black"        : (( 30,  25,  45), "⚫", "Black"),
}

# Recipes: frozenset of two ingredient names → result name + potion name
RECIPES = {
    frozenset(["red",    "blue"])   : ("purple",  "Mystic Brew"),
    frozenset(["red",    "yellow"]) : ("orange",  "Blaze Elixir"),
    frozenset(["blue",   "yellow"]) : ("green",   "Forest Tonic"),
    frozenset(["red",    "white"])  : ("pink",    "Charm Potion"),
    frozenset(["blue",   "white"])  : ("cyan",    "Frost Draught"),
    frozenset(["yellow", "white"])  : ("gold",    "Golden Essence"),
    frozenset(["red",    "green"])  : ("brown",   "Earth Extract"),
    frozenset(["blue",   "purple"]) : ("cyan",    "Shadow Mist"),
    frozenset(["red",    "purple"]) : ("pink",    "Petal Infusion"),
    frozenset(["yellow", "purple"]) : ("brown",   "Ancient Mud"),
    frozenset(["green",  "purple"]) : ("black",   "Void Essence"),
    frozenset(["orange", "blue"])   : ("brown",   "Rustic Potion"),
    frozenset(["orange", "purple"]) : ("brown",   "Dusk Elixir"),
    frozenset(["red",    "black"])  : ("brown",   "Dark Fire"),
    frozenset(["blue",   "black"])  : ("purple",  "Abyss Water"),
    frozenset(["yellow", "orange"]) : ("gold",    "Sun Extract"),
}

# ── Particle system ────────────────────────────────────────────────────────────
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float       # 0-1
    color: tuple
    size: float

def spawn_particles(px, py, color, n=24):
    parts = []
    for _ in range(n):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5.0)
        parts.append(Particle(
            x=px, y=py,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            life=1.0,
            color=color,
            size=random.uniform(4, 9),
        ))
    return parts

# ── Potion sprite ──────────────────────────────────────────────────────────────
class Potion:
    RADIUS = 36

    def __init__(self, name, x, y):
        self.name   = name
        col, emoji, display = POTION_DEFS[name]
        self.color  = col
        self.emoji  = emoji
        self.display= display
        self.x      = float(x)
        self.y      = float(y)
        self.vx     = random.uniform(-0.8, 0.8)
        self.vy     = random.uniform(-0.8, 0.8)
        self.grabbed= False
        self.grab_hand: Optional[int] = None  # hand index
        self.wobble = 0.0
        self.scale  = 1.0
        self.target_scale = 1.0
        self.birth  = time.time()
        self.alive  = True

    def draw(self, surf):
        t = time.time() - self.birth
        bob = math.sin(t * 2.2) * 4
        s = self.scale + math.sin(t * 5) * 0.02
        r = int(self.RADIUS * s)
        cx, cy = int(self.x), int(self.y + bob)

        # Glow
        glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        for i in range(3, 0, -1):
            alpha = 40 // i
            gr = r + i * 8
            col_a = (*self.color, alpha)
            pygame.draw.circle(glow_surf, col_a, (r * 2, r * 2), gr)
        surf.blit(glow_surf, (cx - r * 2, cy - r * 2))

        # Body
        pygame.draw.circle(surf, self.color, (cx, cy), r)

        # Shine
        shine_r = max(4, r // 3)
        shine_pos = (cx - r // 3, cy - r // 3)
        shine_surf = pygame.Surface((shine_r * 2, shine_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(shine_surf, (255, 255, 255, 100), (shine_r, shine_r), shine_r)
        surf.blit(shine_surf, (shine_pos[0] - shine_r, shine_pos[1] - shine_r))

        # Bottle neck
        neck_w = r // 2
        neck_h = r // 2
        neck_rect = pygame.Rect(cx - neck_w // 2, cy - r - neck_h + 4, neck_w, neck_h)
        pygame.draw.rect(surf, self.color, neck_rect, border_radius=4)
        pygame.draw.rect(surf, tuple(min(255, c + 40) for c in self.color),
                         neck_rect, width=2, border_radius=4)

        # Cork
        cork_rect = pygame.Rect(cx - neck_w // 2 - 2, cy - r - neck_h - 4, neck_w + 4, 10)
        pygame.draw.rect(surf, (160, 110, 60), cork_rect, border_radius=3)

        # Outline (thicker when grabbed)
        lw = 3 if self.grabbed else 1
        pygame.draw.circle(surf, WHITE if self.grabbed else GREY, (cx, cy), r, lw)

        # Label
        label = font_sm.render(self.display, True, WHITE)
        surf.blit(label, (cx - label.get_width() // 2, cy + r + 6))

    def update(self):
        if not self.grabbed:
            self.x += self.vx
            self.y += self.vy
            # Bounce off walls
            pad = self.RADIUS + 60
            if self.x < pad:      self.x = pad;        self.vx = abs(self.vx)
            if self.x > W - pad:  self.x = W - pad;    self.vx = -abs(self.vx)
            if self.y < pad:      self.y = pad;         self.vy = abs(self.vy)
            if self.y > H - 120:  self.y = H - 120;    self.vy = -abs(self.vy)
            # Slight drag
            self.vx *= 0.995
            self.vy *= 0.995
        self.scale += (self.target_scale - self.scale) * 0.15

# ── Floating text popup ────────────────────────────────────────────────────────
@dataclass
class Popup:
    text: str
    x: float
    y: float
    life: float = 1.5
    color: tuple = (255, 255, 180)

# ── Hand tracker — MediaPipe Vision Tasks HandLandmarker ───────────────────────
PINCH_THRESHOLD = 0.06   # normalised distance (index-tip ↔ thumb-tip)

# Landmark indices (same numbering as classic MediaPipe Hands)
IDX_THUMB_TIP = 4
IDX_INDEX_TIP = 8

class HandTracker:
    """
    Uses mp.tasks.vision.HandLandmarker in LIVE_STREAM mode.
    Results arrive in a callback and are stored thread-safely in instance vars.
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        self.pinch_positions: dict[int, tuple[float, float]] = {}
        self.pinch_active:    dict[int, bool]                = {}
        self.cam_frame = None          # latest BGR frame (for preview)
        self._latest_result: Optional[HandLandmarkerResult] = None
        self._ts_ms = 0                # monotonic timestamp for API

        # Build HandLandmarker in LIVE_STREAM mode
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # ── Called from the MediaPipe worker thread ────────────────────────────────
    def _on_result(self, result: HandLandmarkerResult, output_image, timestamp_ms):
        self._latest_result = result

    # ── Called every game frame from the main thread ───────────────────────────
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)          # mirror so it feels natural
        self.cam_frame = frame

        # Convert to MediaPipe Image (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonically increasing timestamp required by LIVE_STREAM mode
        self._ts_ms += 1
        self._landmarker.detect_async(mp_image, self._ts_ms)

        # Parse the most recent result (may be one frame behind — that's fine)
        result = self._latest_result
        new_pos:    dict[int, tuple[float, float]] = {}
        new_active: dict[int, bool]                = {}

        if result and result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                thumb = hand_lms[IDX_THUMB_TIP]
                index = hand_lms[IDX_INDEX_TIP]

                # Normalised coords (0-1); x is already mirror-flipped above
                mx = (thumb.x + index.x) / 2
                my = (thumb.y + index.y) / 2

                gx = mx * W
                gy = my * H

                dist   = math.hypot(thumb.x - index.x, thumb.y - index.y)
                active = dist < PINCH_THRESHOLD

                new_pos[i]    = (gx, gy)
                new_active[i] = active

        self.pinch_positions = new_pos
        self.pinch_active    = new_active

    def get_cam_surface(self, target_w=220, target_h=124):
        if self.cam_frame is None:
            return None
        small = cv2.resize(self.cam_frame, (target_w, target_h))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def get_full_frame_surface(self):
        """Return the full camera frame as a pygame surface."""
        if self.cam_frame is None:
            return None
        rgb = cv2.cvtColor(self.cam_frame, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def release(self):
        self._landmarker.close()
        self.cap.release()

# ── Main game ──────────────────────────────────────────────────────────────────
class Game:
    def __init__(self):
        self.tracker   = HandTracker()
        self.potions: list[Potion] = []
        self.particles: list[Particle] = []
        self.popups: list[Popup] = []
        self.score     = 0
        self.combos    = []   # list of (result_name, recipe_name)
        self.discovered= set()
        self.prev_pinch= {}   # hand_idx → bool (for edge detection)
        self.spawn_timer = 0
        self._spawn_initial()

    def _spawn_initial(self):
        bases = ["red", "blue", "yellow", "white", "white", "yellow"]
        for i, name in enumerate(bases):
            x = 200 + i * 250
            y = random.randint(200, 450)
            self.potions.append(Potion(name, x, y))

    def _spawn_potion(self, name, x, y):
        p = Potion(name, x, y)
        p.vx = random.uniform(-1.5, 1.5)
        p.vy = random.uniform(-2, -0.5)
        self.potions.append(p)

    def _try_mix(self, a: Potion, b: Potion):
        key = frozenset([a.name, b.name])
        if key in RECIPES:
            result_name, recipe_name = RECIPES[key]
            mx = (a.x + b.x) / 2
            my = (a.y + b.y) / 2
            col = POTION_DEFS[result_name][0]
            self.particles += spawn_particles(mx, my, col, 32)
            a.alive = False
            b.alive = False
            self._spawn_potion(result_name, mx, my)
            self.score += 10
            self.popups.append(Popup(f"✨ {recipe_name}!", mx, my - 60,
                                     color=(255, 220, 80)))
            if recipe_name not in self.discovered:
                self.discovered.add(recipe_name)
                self.score += 20
                self.popups.append(Popup("NEW RECIPE! +20", mx, my - 100,
                                         color=(80, 255, 180)))
            self.combos.append((result_name, recipe_name))
        else:
            # No recipe — bounce away
            self.popups.append(Popup("No recipe...", (a.x+b.x)/2, (a.y+b.y)/2 - 40,
                                     color=(200, 140, 140)))
            a.vx = -2; b.vx = 2

    def update(self):
        self.tracker.update()
        positions = self.tracker.pinch_positions
        active    = self.tracker.pinch_active

        # Grab / release
        for hi, (gx, gy) in positions.items():
            is_pinching = active.get(hi, False)
            was_pinching = self.prev_pinch.get(hi, False)

            if is_pinching:
                # Move grabbed potion
                for p in self.potions:
                    if p.grab_hand == hi:
                        p.x = gx
                        p.y = gy
                        p.target_scale = 1.2

                # New pinch — find closest potion to grab
                if not was_pinching:
                    closest = None
                    best_d  = Potion.RADIUS * 1.8
                    for p in self.potions:
                        if not p.grabbed:
                            d = math.hypot(p.x - gx, p.y - gy)
                            if d < best_d:
                                best_d = d
                                closest = p
                    if closest:
                        closest.grabbed   = True
                        closest.grab_hand = hi
                        closest.vx = closest.vy = 0
            else:
                # Release
                for p in self.potions:
                    if p.grab_hand == hi:
                        p.grabbed      = False
                        p.grab_hand    = None
                        p.target_scale = 1.0
                        p.vx = random.uniform(-1, 1)
                        p.vy = random.uniform(-1, 1)

        self.prev_pinch = {hi: active.get(hi, False) for hi in positions}

        # Check mixing — two grabbed potions from different hands that are close
        grabbed = [p for p in self.potions if p.grabbed]
        for i in range(len(grabbed)):
            for j in range(i + 1, len(grabbed)):
                a, b = grabbed[i], grabbed[j]
                if a.grab_hand != b.grab_hand:
                    d = math.hypot(a.x - b.x, a.y - b.y)
                    if d < Potion.RADIUS * 2.2:
                        self._try_mix(a, b)

        # Also allow mixing when one is held and one is free but close
        for a in [p for p in self.potions if p.grabbed]:
            for b in [p for p in self.potions if not p.grabbed and p is not a]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < Potion.RADIUS * 1.8:
                    self._try_mix(a, b)
                    break

        # Update potions
        for p in self.potions:
            p.update()
        self.potions = [p for p in self.potions if p.alive]

        # Respawn if too few base potions
        self.spawn_timer += 1
        if self.spawn_timer > 180 and len(self.potions) < 3:
            self.spawn_timer = 0
            name = random.choice(["red", "blue", "yellow", "white"])
            x = random.randint(200, W - 200)
            y = random.randint(180, 380)
            self._spawn_potion(name, x, y)

        # Particles
        for p in self.particles:
            p.x  += p.vx;  p.y  += p.vy
            p.vy += 0.12
            p.vx *= 0.97;  p.vy *= 0.97
            p.life -= 0.025
        self.particles = [p for p in self.particles if p.life > 0]

        # Popups
        for pp in self.popups:
            pp.y   -= 0.8
            pp.life -= 1 / 60
        self.popups = [pp for pp in self.popups if pp.life > 0]

    def draw(self):
        # Background — full camera frame
        cam_surf = self.tracker.get_full_frame_surface()
        if cam_surf:
            screen.blit(cam_surf, (0, 0))
        else:
            screen.fill(BG)

        # Camera overlay (slight darkening for UI visibility)
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 40))
        screen.blit(overlay, (0, 0))

        # Potions
        for p in self.potions:
            p.draw(screen)

        # Particles
        for p in self.particles:
            alpha = int(p.life * 255)
            ps = pygame.Surface((int(p.size)*2, int(p.size)*2), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*p.color, alpha), (int(p.size), int(p.size)), int(p.size))
            screen.blit(ps, (int(p.x - p.size), int(p.y - p.size)))

        # Pinch cursors
        for hi, (gx, gy) in self.tracker.pinch_positions.items():
            is_pinching = self.tracker.pinch_active.get(hi, False)
            col = (255, 220, 80) if is_pinching else (180, 180, 220)
            r   = 14 if is_pinching else 10
            pygame.draw.circle(screen, col, (int(gx), int(gy)), r, 3)
            pygame.draw.circle(screen, (*col, 80), (int(gx), int(gy)), r + 6, 1)
            label = font_sm.render("pinching" if is_pinching else "open", True, col)
            screen.blit(label, (int(gx) + 16, int(gy) - 8))

        # Popups
        for pp in self.popups:
            alpha = min(255, int(pp.life * 400))
            surf  = font_md.render(pp.text, True, pp.color)
            a_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            a_surf.blit(surf, (0, 0))
            a_surf.set_alpha(alpha)
            screen.blit(a_surf, (int(pp.x - surf.get_width()//2), int(pp.y)))

        # HUD — top bar
        pygame.draw.rect(screen, PANEL, (0, 0, W, 54))
        title = font_lg.render("✨ Potion Mixer", True, (200, 160, 255))
        screen.blit(title, (16, 12))
        score_txt = font_lg.render(f"Score: {self.score}", True, (255, 220, 80))
        screen.blit(score_txt, (W - score_txt.get_width() - 16, 12))
        disc_txt = font_sm.render(
            f"Recipes discovered: {len(self.discovered)} / {len(RECIPES)}", True, GREY)
        screen.blit(disc_txt, (W//2 - disc_txt.get_width()//2, 18))

        # Camera preview removed — using full camera overlay

        # Instruction bar (bottom)
        pygame.draw.rect(screen, PANEL, (0, H - 36, W, 36))
        hint = font_sm.render(
            "Pinch (index + thumb) to grab a potion · Bring two potions together to mix · Scroll wheel skips camera debug",
            True, GREY)
        screen.blit(hint, (W//2 - hint.get_width()//2, H - 26))

        # Last 4 combos sidebar
        if self.combos:
            sidebar_x = W - 230
            sidebar_y = 70
            pygame.draw.rect(screen, PANEL,
                             (sidebar_x - 8, sidebar_y - 8, 222, min(4, len(self.combos)) * 30 + 36),
                             border_radius=8)
            head = font_sm.render("Recent mixes", True, (180, 160, 220))
            screen.blit(head, (sidebar_x, sidebar_y))
            for i, (rname, recipe) in enumerate(self.combos[-4:][::-1]):
                col = POTION_DEFS[rname][0]
                dot_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
                pygame.draw.circle(dot_surf, col, (6, 6), 6)
                screen.blit(dot_surf, (sidebar_x, sidebar_y + 26 + i * 28))
                txt = font_sm.render(recipe, True, WHITE)
                screen.blit(txt, (sidebar_x + 16, sidebar_y + 24 + i * 28))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.tracker.release()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.tracker.release()
                    pygame.quit()
                    sys.exit()
                # Mouse fallback for testing without camera
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    for p in self.potions:
                        if math.hypot(p.x - mx, p.y - my) < Potion.RADIUS + 10:
                            p.grabbed = True
                            p.grab_hand = 99
                            p.vx = p.vy = 0
                            break
                if event.type == pygame.MOUSEBUTTONUP:
                    for p in self.potions:
                        if p.grab_hand == 99:
                            p.grabbed = False
                            p.grab_hand = None
                            p.target_scale = 1.0
                            p.vx = random.uniform(-1, 1)
                            p.vy = random.uniform(-1, 1)
                if event.type == pygame.MOUSEMOTION:
                    mx, my = pygame.mouse.get_pos()
                    for p in self.potions:
                        if p.grab_hand == 99:
                            p.x = mx; p.y = my

            self.update()
            self.draw()
            clock.tick(60)


if __name__ == "__main__":
    print("Starting Potion Mixer...")
    print("  Pinch index finger + thumb to grab potions")
    print("  Bring two potions together to mix them")
    print("  Press ESC to quit\n")
    game = Game()
    game.run()