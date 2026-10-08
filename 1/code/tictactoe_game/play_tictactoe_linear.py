from __future__ import annotations

import argparse
import sys
import pygame

from tictactoe_linear_agent import (
    LinearTicTacToeAgent,
    apply_move,
    check_winner,
    is_full,
    legal_moves,
    other_player,
)

WIDTH, HEIGHT = 600, 700
BOARD_PIXELS = 600
CELL = BOARD_PIXELS // 3
FPS = 60


class TicTacToeGUI:
    def __init__(self, agent: LinearTicTacToeAgent, human_player: int = 1, ai_vs_ai: bool = False):
        pygame.init()
        pygame.display.set_caption("Tic-tac-toe Linear Agent")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.agent = agent
        self.human_player = human_player
        self.ai_vs_ai = ai_vs_ai
        self.font = pygame.font.SysFont(None, 42)
        self.big_font = pygame.font.SysFont(None, 96)
        self.reset()

    def reset(self) -> None:
        self.board = [0] * 9
        self.player = 1
        self.game_over = False
        self.winner = 0
        self.status_text = "X to move"

    def handle_click(self, pos) -> None:
        if self.game_over or self.ai_vs_ai or self.player != self.human_player:
            return
        x, y = pos
        if y >= BOARD_PIXELS:
            return
        c = x // CELL
        r = y // CELL
        idx = r * 3 + c
        if idx in legal_moves(self.board):
            self.board = apply_move(self.board, idx, self.player)
            self.finish_or_continue()

    def finish_or_continue(self) -> None:
        self.winner = check_winner(self.board)
        if self.winner != 0:
            self.game_over = True
            self.status_text = f"{'X' if self.winner == 1 else 'O'} wins! Press R to restart."
            return
        if is_full(self.board):
            self.game_over = True
            self.status_text = "Draw! Press R to restart."
            return
        self.player = other_player(self.player)
        self.status_text = f"{'X' if self.player == 1 else 'O'} to move"

    def ai_step(self) -> None:
        if self.game_over:
            return
        if self.ai_vs_ai or self.player != self.human_player:
            mv = self.agent.choose_move(self.board, self.player, training=False)
            self.board = apply_move(self.board, mv, self.player)
            self.finish_or_continue()

    def draw_board(self) -> None:
        self.screen.fill((245, 245, 245))
        for i in range(1, 3):
            pygame.draw.line(self.screen, (50, 50, 50), (i * CELL, 0), (i * CELL, BOARD_PIXELS), 4)
            pygame.draw.line(self.screen, (50, 50, 50), (0, i * CELL), (BOARD_PIXELS, i * CELL), 4)

        for i, val in enumerate(self.board):
            r = i // 3
            c = i % 3
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2
            if val == 1:
                text = self.big_font.render("X", True, (20, 20, 20))
                rect = text.get_rect(center=(cx, cy))
                self.screen.blit(text, rect)
            elif val == -1:
                text = self.big_font.render("O", True, (20, 20, 20))
                rect = text.get_rect(center=(cx, cy))
                self.screen.blit(text, rect)

        pygame.draw.rect(self.screen, (230, 230, 230), (0, BOARD_PIXELS, WIDTH, HEIGHT - BOARD_PIXELS))
        status = self.font.render(self.status_text, True, (20, 20, 20))
        self.screen.blit(status, (20, BOARD_PIXELS + 20))
        self.screen.blit(self.font.render("R: restart", True, (40, 40, 40)), (20, BOARD_PIXELS + 70))
        self.screen.blit(self.font.render("ESC: quit", True, (40, 40, 40)), (20, BOARD_PIXELS + 110))
        if not self.ai_vs_ai:
            side = "X" if self.human_player == 1 else "O"
            self.screen.blit(self.font.render(f"Human = {side}", True, (40, 40, 40)), (280, BOARD_PIXELS + 70))
        else:
            self.screen.blit(self.font.render("AI vs AI mode", True, (40, 40, 40)), (280, BOARD_PIXELS + 70))
        pygame.display.flip()

    def run(self) -> None:
        ai_delay_ms = 250
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        raise SystemExit
                    elif event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            if not self.game_over and (self.ai_vs_ai or self.player != self.human_player):
                pygame.time.delay(ai_delay_ms)
                self.ai_step()

            self.draw_board()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="tictactoe_linear_weights.pkl")
    parser.add_argument("--human", choices=["X", "O"], default="X")
    parser.add_argument("--ai_vs_ai", action="store_true")
    args = parser.parse_args()

    try:
        agent = LinearTicTacToeAgent.load(args.weights)
    except FileNotFoundError:
        print(f"Could not find weights file: {args.weights}")
        print("Train first:")
        print("python train_tictactoe_linear.py --episodes 20000")
        sys.exit(1)

    human_player = 1 if args.human.upper() == "X" else -1
    app = TicTacToeGUI(agent=agent, human_player=human_player, ai_vs_ai=args.ai_vs_ai)
    app.run()


if __name__ == "__main__":
    main()
