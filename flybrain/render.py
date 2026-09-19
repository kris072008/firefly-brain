"""pygame front end."""
from __future__ import annotations

import math

import pygame

from .game import H_ARENA, SENSE_RANGE, W_ARENA, BrainController, World

BG = (12, 12, 18)
FG = (235, 235, 240)
SUGAR = (120, 220, 140)
BITTER = (230, 90, 110)
FLYC = (120, 180, 255)


def play(brain, seed: int = 0, fps: int = 20, cell_type=None,
         invert: bool = False, swap: bool = False):
    pygame.init()
    screen = pygame.display.set_mode((W_ARENA, H_ARENA + 92))
    pygame.display.set_caption("FlyWire connectome plays a game")
    try:
        font = pygame.font.SysFont("monospace", 13)
    except Exception:          # some pygame builds ship without the font module
        font = None
        print("note: pygame has no font module, running without the HUD text")
    clock = pygame.time.Clock()
    world = World(seed, swap=swap)
    ctrl = BrainController(brain, cell_type=cell_type, invert=invert)
    print(f"steering on {ctrl.name}: "
          f"{ctrl.dn_left.size} left, {ctrl.dn_right.size} right")
    dt = 1.0 / fps
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        turn, l, r = ctrl.tick(world.sense(), dt)
        keys = pygame.key.get_pressed()
        manual = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
        if manual:                      # hold an arrow key to steer it yourself
            turn = -2.5 if keys[pygame.K_LEFT] else 2.5
        world.advance(turn, dt)

        screen.fill(BG)
        f = world.fly
        pygame.draw.circle(screen, (30, 30, 42), (int(f.x), int(f.y)), int(SENSE_RANGE), 1)
        for sx, sy in world.food:
            pygame.draw.circle(screen, SUGAR, (int(sx), int(sy)), 7)
        for sx, sy in world.poison:
            pygame.draw.circle(screen, BITTER, (int(sx), int(sy)), 7)
        tip = (f.x + math.cos(f.heading) * 16, f.y + math.sin(f.heading) * 16)
        left = (f.x + math.cos(f.heading + 2.5) * 10, f.y + math.sin(f.heading + 2.5) * 10)
        right = (f.x + math.cos(f.heading - 2.5) * 10, f.y + math.sin(f.heading - 2.5) * 10)
        pygame.draw.polygon(screen, FLYC, [tip, left, right])

        active = int((brain.spike_counts > 0).sum())
        lines = [
            f"sugar eaten {f.eaten:3d}   bitter hit {f.poisoned:3d}   t={brain.t:6.1f}s"
            + ("   [MANUAL]" if manual else "   [brain driving]"),
            f"{ctrl.name:<7} left {l:6.1f} Hz  right {r:6.1f} Hz  turn {turn:+.2f} rad/s",
            f"neurons that have fired: {active:,} / {brain.n:,}",
            "drive (Hz)  " + "  ".join(
                f"{k.replace('_', ' ')} {ctrl.rates.get(k, 0.0):3.0f}"
                for k in ("sugar_left", "sugar_right",
                          "bitter_left", "bitter_right")),
        ]
        if font is not None:
            for i, s in enumerate(lines):
                screen.blit(font.render(s, True, FG), (12, H_ARENA + 6 + i * 19))
        elif int(brain.t * 2) % 4 == 0:
            print(" | ".join(lines))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
