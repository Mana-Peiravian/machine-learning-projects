import argparse
import sys
import time
import pygame

from snake_linear_agent import (
    SnakeGame,
    LinearSnakeAgent,
    ACTIONS,
    UP, DOWN, LEFT, RIGHT,
    add_pos,
    BOARD_SIZE,
)

CELL = 28
MARGIN = 60
WIDTH = BOARD_SIZE * CELL
HEIGHT = BOARD_SIZE * CELL + MARGIN

WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GREEN = (46, 204, 113)
DARK_GREEN = (39, 174, 96)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
GRAY = (210, 210, 210)
YELLOW = (241, 196, 15)

KEY_TO_DIR = {
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
}

def draw(game, screen, font, manual):
    screen.fill(WHITE)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            rect = pygame.Rect(x * CELL, y * CELL + MARGIN, CELL, CELL)
            pygame.draw.rect(screen, GRAY, rect, 1)

    for (x, y) in game.obstacles:
        rect = pygame.Rect(x * CELL, y * CELL + MARGIN, CELL, CELL)
        pygame.draw.rect(screen, BLUE, rect)

    fx, fy = game.food
    pygame.draw.rect(screen, RED, pygame.Rect(fx * CELL, fy * CELL + MARGIN, CELL, CELL))

    for i, (x, y) in enumerate(game.snake):
        color = DARK_GREEN if i == 0 else GREEN
        pygame.draw.rect(screen, color, pygame.Rect(x * CELL, y * CELL + MARGIN, CELL, CELL))

    mode = "MANUAL" if manual else "AGENT"
    status = f"Mode: {mode}   Score: {game.score}   Steps: {game.steps}"
    if game.terminal:
        status += f"   Terminal: {game.death_cause}"
    text = font.render(status, True, BLACK)
    screen.blit(text, (10, 15))
    pygame.display.flip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="snake_linear_weights.pkl")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - Improved Linear Value Agent")
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    game = SnakeGame()
    agent = None if args.manual else LinearSnakeAgent.load(args.weights)

    running = True
    pending_dir = game.direction

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif args.manual and event.type == pygame.KEYDOWN and event.key in KEY_TO_DIR:
                pending_dir = KEY_TO_DIR[event.key]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()
                pending_dir = game.direction

        if not game.terminal:
            if args.manual:
                action = pending_dir
                if action not in game.legal_actions():
                    action = game.direction
            else:
                action = agent.choose_action(game, training=False)
            game.step(action)

        draw(game, screen, font, args.manual)
        clock.tick(args.fps)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
