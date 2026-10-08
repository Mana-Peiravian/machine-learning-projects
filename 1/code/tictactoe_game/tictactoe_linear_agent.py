from __future__ import annotations
"""
Linear value-function approximation agent for Tic-tac-toe.

We encode each board from the CURRENT player's perspective:
    +1 = current player's marks
    -1 = opponent's marks
     0 = empty

For non-terminal states:
    V_train(s_t) = -V_hat(s_{t+1})
because after the current player moves, it becomes the opponent's turn.

Terminal targets:
    win  = +100
    lose = -100
    draw = 0
"""

import math
import pickle
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

WIN_REWARD = 100.0
LOSE_REWARD = -100.0
DRAW_REWARD = 0.0

Line = Tuple[int, int, int]
WIN_LINES: List[Line] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def other_player(player: int) -> int:
    return -player


def check_winner(board: Sequence[int]) -> int:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def is_full(board: Sequence[int]) -> bool:
    return all(v != 0 for v in board)


def is_terminal(board: Sequence[int]) -> bool:
    return check_winner(board) != 0 or is_full(board)


def legal_moves(board: Sequence[int]) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board: Sequence[int], move: int, player: int) -> List[int]:
    new_board = list(board)
    new_board[move] = player
    return new_board


def random_start_state(max_random_moves: int = 5, rng: Optional[random.Random] = None) -> Tuple[List[int], int]:
    rng = rng or random
    board = [0] * 9
    player = 1
    num_moves = rng.randint(0, max_random_moves)

    for _ in range(num_moves):
        moves = legal_moves(board)
        if not moves:
            break
        mv = rng.choice(moves)
        board = apply_move(board, mv, player)
        if is_terminal(board):
            return random_start_state(max_random_moves=max_random_moves, rng=rng)
        player = other_player(player)

    return board, player


def perspective_board(board: Sequence[int], player: int) -> np.ndarray:
    arr = np.array(board, dtype=np.float64)
    return arr * float(player)


def count_open_two(rel_board: np.ndarray, for_me: bool = True) -> int:
    total = 0
    target = 1 if for_me else -1
    for line in WIN_LINES:
        vals = [rel_board[i] for i in line]
        if vals.count(target) == 2 and vals.count(0) == 1:
            total += 1
    return total


def count_open_one(rel_board: np.ndarray, for_me: bool = True) -> int:
    total = 0
    target = 1 if for_me else -1
    for line in WIN_LINES:
        vals = [rel_board[i] for i in line]
        if vals.count(target) == 1 and vals.count(0) == 2:
            total += 1
    return total


def count_forks_after_move(board: Sequence[int], player: int) -> int:
    total = 0
    for mv in legal_moves(board):
        b2 = apply_move(board, mv, player)
        rel = perspective_board(b2, player)
        if count_open_two(rel, for_me=True) >= 2:
            total += 1
    return total


def immediate_winning_moves(board: Sequence[int], player: int) -> int:
    total = 0
    for mv in legal_moves(board):
        b2 = apply_move(board, mv, player)
        if check_winner(b2) == player:
            total += 1
    return total


def extract_features(board: Sequence[int], player: int) -> np.ndarray:
    rel = perspective_board(board, player)
    winner = check_winner(board)
    empties = sum(1 for v in board if v == 0)

    my_marks = float(np.sum(rel == 1))
    opp_marks = float(np.sum(rel == -1))

    center_me = 1.0 if rel[4] == 1 else 0.0
    center_opp = 1.0 if rel[4] == -1 else 0.0

    corners = [0, 2, 6, 8]
    edges = [1, 3, 5, 7]
    my_corners = float(sum(1 for i in corners if rel[i] == 1))
    opp_corners = float(sum(1 for i in corners if rel[i] == -1))
    my_edges = float(sum(1 for i in edges if rel[i] == 1))
    opp_edges = float(sum(1 for i in edges if rel[i] == -1))

    my_open_two = float(count_open_two(rel, True))
    opp_open_two = float(count_open_two(rel, False))
    my_open_one = float(count_open_one(rel, True))
    opp_open_one = float(count_open_one(rel, False))

    my_immediate_wins = float(immediate_winning_moves(board, player))
    opp_immediate_wins = float(immediate_winning_moves(board, other_player(player)))

    my_forks = float(count_forks_after_move(board, player))
    opp_forks = float(count_forks_after_move(board, other_player(player)))

    my_open_lines = 0.0
    opp_open_lines = 0.0
    for line in WIN_LINES:
        vals = [rel[i] for i in line]
        if -1 not in vals:
            my_open_lines += 1.0
        if 1 not in vals:
            opp_open_lines += 1.0

    terminal_win = 1.0 if winner == player else 0.0
    terminal_loss = 1.0 if winner == other_player(player) else 0.0
    terminal_draw = 1.0 if winner == 0 and empties == 0 else 0.0

    return np.array([
        1.0,
        my_marks / 5.0,
        opp_marks / 5.0,
        empties / 9.0,
        center_me,
        center_opp,
        my_corners / 4.0,
        opp_corners / 4.0,
        my_edges / 4.0,
        opp_edges / 4.0,
        my_open_two / 3.0,
        opp_open_two / 3.0,
        my_open_one / 8.0,
        opp_open_one / 8.0,
        my_immediate_wins / 3.0,
        opp_immediate_wins / 3.0,
        my_forks / 4.0,
        opp_forks / 4.0,
        my_open_lines / 8.0,
        opp_open_lines / 8.0,
        (my_open_two - opp_open_two) / 3.0,
        (my_forks - opp_forks) / 4.0,
        (my_open_lines - opp_open_lines) / 8.0,
        terminal_win,
        terminal_loss,
        terminal_draw,
    ], dtype=np.float64)


@dataclass
class LinearTicTacToeAgent:
    alpha: float = 0.02
    epsilon: float = 0.10
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    rng_seed: int = 42

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)
        dim = extract_features([0] * 9, 1).shape[0]
        self.w = np.zeros(dim, dtype=np.float64)

    def reset_weights(self) -> None:
        self.w[:] = 0.0

    def value(self, board: Sequence[int], player: int) -> float:
        x = extract_features(board, player)
        return float(np.dot(self.w, x))

    def choose_greedy_move(self, board: Sequence[int], player: int) -> int:
        moves = legal_moves(board)
        best_move = moves[0]
        best_value = -math.inf

        for mv in moves:
            b2 = apply_move(board, mv, player)
            if check_winner(b2) == player:
                v = WIN_REWARD
            elif is_full(b2):
                v = DRAW_REWARD
            else:
                v = -self.value(b2, other_player(player))
            if v > best_value:
                best_value = v
                best_move = mv
        return best_move

    def choose_move(self, board: Sequence[int], player: int, training: bool = False) -> int:
        moves = legal_moves(board)
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        return self.choose_greedy_move(board, player)

    def update_from_state_target(self, board: Sequence[int], player: int, target: float) -> float:
        x = extract_features(board, player)
        pred = float(np.dot(self.w, x))
        error = target - pred
        self.w += self.alpha * error * x
        return error

    def train_one_episode(self, random_start: bool = True) -> dict:
        if random_start:
            board, player = random_start_state(rng=self.rng)
        else:
            board, player = [0] * 9, 1

        trajectory = []

        while True:
            current_board = list(board)
            current_player = player

            mv = self.choose_move(current_board, current_player, training=True)
            next_board = apply_move(current_board, mv, current_player)
            winner = check_winner(next_board)

            if winner == current_player:
                target = WIN_REWARD
                self.update_from_state_target(current_board, current_player, target)
                trajectory.append((current_board, current_player))
                result = current_player
                break

            if is_full(next_board):
                target = DRAW_REWARD
                self.update_from_state_target(current_board, current_player, target)
                trajectory.append((current_board, current_player))
                result = 0
                break

            next_player = other_player(current_player)
            target = -self.value(next_board, next_player)
            self.update_from_state_target(current_board, current_player, target)
            trajectory.append((current_board, current_player))

            board = next_board
            player = next_player

        if result != 0 and len(trajectory) > 1:
            for tb, tp in reversed(trajectory[:-1]):
                target = WIN_REWARD if tp == result else LOSE_REWARD
                self.update_from_state_target(tb, tp, target)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"result": result, "epsilon": self.epsilon}

    def self_play_training(self, episodes: int = 10000, random_start: bool = True) -> dict:
        stats = {"X": 0, "O": 0, "draw": 0}
        for _ in range(episodes):
            info = self.train_one_episode(random_start=random_start)
            if info["result"] == 1:
                stats["X"] += 1
            elif info["result"] == -1:
                stats["O"] += 1
            else:
                stats["draw"] += 1
        return stats

    def play_vs_random(self, games: int = 1000, as_player: int = 1) -> dict:
        stats = {"win": 0, "lose": 0, "draw": 0}
        for _ in range(games):
            board = [0] * 9
            player = 1
            while True:
                if player == as_player:
                    mv = self.choose_move(board, player, training=False)
                else:
                    mv = self.rng.choice(legal_moves(board))
                board = apply_move(board, mv, player)
                winner = check_winner(board)
                if winner == as_player:
                    stats["win"] += 1
                    break
                if winner == other_player(as_player):
                    stats["lose"] += 1
                    break
                if is_full(board):
                    stats["draw"] += 1
                    break
                player = other_player(player)
        return stats

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.w,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "rng_seed": self.rng_seed,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LinearTicTacToeAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha=data.get("alpha", 0.02),
            epsilon=data.get("epsilon", 0.0),
            epsilon_min=data.get("epsilon_min", 0.0),
            epsilon_decay=data.get("epsilon_decay", 1.0),
            rng_seed=data.get("rng_seed", 42),
        )
        agent.w = np.array(data["weights"], dtype=np.float64)
        return agent


def mark_to_char(v: int) -> str:
    return "X" if v == 1 else ("O" if v == -1 else " ")


def board_to_string(board: Sequence[int]) -> str:
    rows = []
    for r in range(3):
        row = [mark_to_char(board[3 * r + c]) for c in range(3)]
        rows.append(" " + " | ".join(row) + " ")
    return "\n-----------\n".join(rows)


if __name__ == "__main__":
    agent = LinearTicTacToeAgent()
    print("Feature dimension:", len(agent.w))
    b, p = random_start_state()
    print(board_to_string(b))
    print("Player to move:", "X" if p == 1 else "O")
    print("Estimated value:", agent.value(b, p))
