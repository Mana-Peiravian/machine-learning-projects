import math
import random
import pickle
from collections import deque, Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

BOARD_SIZE = 20
INITIAL_SNAKE_LEN = 3
OBSTACLE_BLOCKS = 4
OBSTACLE_SIZE = 2

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}
OPPOSITE = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}


def add_pos(a, b):
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class StepResult:
    reward: float
    terminal: bool
    cause: Optional[str] = None
    ate_food: bool = False


class SnakeGame:
    """
    Snake environment for linear value-function learning.
    Board coordinates are (x, y), with 0 <= x,y < 20.
    """

    def __init__(self, board_size: int = BOARD_SIZE, seed: Optional[int] = None):
        self.board_size = board_size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.direction = self.rng.choice(ACTIONS)
        self.snake = self._random_initial_snake()
        self.obstacles = self._random_obstacles()
        self.food = self._random_empty_cell()
        self.score = 0
        self.steps = 0
        self.terminal = False
        self.death_cause = None
        self.visits = Counter()
        self.visits[self.head] += 1
        return self

    @property
    def head(self):
        return self.snake[0]

    def clone(self):
        g = SnakeGame(self.board_size)
        g.rng = self.rng
        g.direction = self.direction
        g.snake = list(self.snake)
        g.obstacles = set(self.obstacles)
        g.food = self.food
        g.score = self.score
        g.steps = self.steps
        g.terminal = self.terminal
        g.death_cause = self.death_cause
        g.visits = Counter(self.visits)
        return g

    def _inside(self, p):
        return 0 <= p[0] < self.board_size and 0 <= p[1] < self.board_size

    def _random_initial_snake(self):
        while True:
            d = self.rng.choice(ACTIONS)
            # place head away from borders so len-3 fits
            x = self.rng.randint(2, self.board_size - 3)
            y = self.rng.randint(2, self.board_size - 3)
            head = (x, y)
            body = [head]
            ok = True
            cur = head
            back = OPPOSITE[d]
            for _ in range(INITIAL_SNAKE_LEN - 1):
                cur = add_pos(cur, back)
                if not self._inside(cur):
                    ok = False
                    break
                body.append(cur)
            if ok:
                self.direction = d
                return body

    def _random_obstacles(self):
        occupied = set(self.snake)
        obstacles = set()
        attempts = 0
        while len(obstacles) < OBSTACLE_BLOCKS * OBSTACLE_SIZE * OBSTACLE_SIZE and attempts < 5000:
            attempts += 1
            x = self.rng.randint(0, self.board_size - OBSTACLE_SIZE)
            y = self.rng.randint(0, self.board_size - OBSTACLE_SIZE)
            block = {(x + dx, y + dy) for dx in range(OBSTACLE_SIZE) for dy in range(OBSTACLE_SIZE)}
            if block & occupied or block & obstacles:
                continue
            obstacles |= block
        if len(obstacles) < OBSTACLE_BLOCKS * OBSTACLE_SIZE * OBSTACLE_SIZE:
            raise RuntimeError("Could not place obstacles without overlap.")
        return obstacles

    def _random_empty_cell(self):
        occupied = set(self.snake) | set(self.obstacles)
        empty = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if (x, y) not in occupied]
        return self.rng.choice(empty)

    def legal_actions(self):
        # Prevent immediate reversal
        acts = []
        for a in ACTIONS:
            if len(self.snake) > 1 and a == OPPOSITE[self.direction]:
                continue
            acts.append(a)
        return acts

    def is_blocked(self, pos, grow=False):
        if not self._inside(pos):
            return "wall"
        if pos in self.obstacles:
            return "obstacle"
        body = self.snake[:-1] if not grow else self.snake
        if pos in body:
            return "self"
        return None

    def step(self, action):
        if self.terminal:
            return StepResult(0.0, True, self.death_cause, False)

        self.direction = action
        new_head = add_pos(self.head, action)
        ate_food = (new_head == self.food)
        blocked = self.is_blocked(new_head, grow=ate_food)
        self.steps += 1

        if blocked is not None:
            self.terminal = True
            self.death_cause = blocked
            return StepResult(-1000.0, True, blocked, False)

        self.snake.insert(0, new_head)
        if ate_food:
            self.score += 1
            self.food = self._random_empty_cell()
            reward = 80.0
        else:
            self.snake.pop()
            reward = -1.0

        # small revisit penalty to reduce loops
        self.visits[new_head] += 1
        reward -= 2.0 * max(0, self.visits[new_head] - 1)

        # small dense reward to prefer moving toward food
        reward += 2.0 / (1 + manhattan(new_head, self.food))

        # allow very long episodes but stop pure stagnation
        max_steps = 200 + 40 * self.score
        if self.steps >= max_steps:
            self.terminal = True
            self.death_cause = "timeout"
            reward -= 300.0
            return StepResult(reward, True, "timeout", ate_food)

        return StepResult(reward, False, None, ate_food)


class LinearSnakeAgent:
    """
    Value-function approximator: V_hat(s) = w^T x(s)
    Uses one-step bootstrapping and greedy action selection.
    """

    def __init__(self, alpha=0.01, gamma=0.95, epsilon=0.10, seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.feature_names = [
            "bias",
            "food_up",
            "food_down",
            "food_left",
            "food_right",
            "danger_straight",
            "danger_left",
            "danger_right",
            "danger_back",
            "dist_food_norm",
            "closer_to_food_if_forward",
            "closer_to_food_if_left",
            "closer_to_food_if_right",
            "free_neighbors_norm",
            "flood_fill_ratio",
            "tail_reachable",
            "food_reachable",
            "corridor_penalty",
            "length_norm",
            "head_x_norm",
            "head_y_norm",
            "near_wall",
            "visited_penalty",
            "can_eat_next",
            "open_straight_3",
            "open_left_3",
            "open_right_3",
        ]
        self.w = np.zeros(len(self.feature_names), dtype=np.float64)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weights": self.w,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "feature_names": self.feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(alpha=data.get("alpha", 0.01),
                    gamma=data.get("gamma", 0.95),
                    epsilon=data.get("epsilon", 0.0))
        agent.w = np.array(data["weights"], dtype=np.float64)
        if "feature_names" in data and len(data["feature_names"]) == len(agent.w):
            agent.feature_names = data["feature_names"]
        return agent

    def left_of(self, direction):
        dx, dy = direction
        return (dy, -dx)

    def right_of(self, direction):
        dx, dy = direction
        return (-dy, dx)

    def _ray_free_steps(self, game: SnakeGame, direction, max_depth=3):
        p = game.head
        steps = 0
        for _ in range(max_depth):
            p = add_pos(p, direction)
            if game.is_blocked(p) is not None:
                break
            steps += 1
        return steps / max_depth

    def _neighbors(self, p):
        return [add_pos(p, a) for a in ACTIONS]

    def _bfs_area(self, game: SnakeGame, start, allow_tail=True, target=None):
        """
        Flood-fill from start to estimate free space and optional reachability.
        """
        if game.is_blocked(start) is not None:
            return 0, False
        blocked = set(game.obstacles)
        body = set(game.snake[:-1] if allow_tail else game.snake)
        blocked |= body

        q = deque([start])
        seen = {start}
        found_target = (start == target)
        while q:
            p = q.popleft()
            for n in self._neighbors(p):
                if not game._inside(n):
                    continue
                if n in seen:
                    continue
                if n in blocked:
                    continue
                seen.add(n)
                if target is not None and n == target:
                    found_target = True
                q.append(n)
        return len(seen), found_target

    def features(self, game: SnakeGame):
        head = game.head
        food = game.food
        d = game.direction
        left = self.left_of(d)
        right = self.right_of(d)
        back = OPPOSITE[d]

        straight_cell = add_pos(head, d)
        left_cell = add_pos(head, left)
        right_cell = add_pos(head, right)
        back_cell = add_pos(head, back)

        food_up = 1.0 if food[1] < head[1] else 0.0
        food_down = 1.0 if food[1] > head[1] else 0.0
        food_left = 1.0 if food[0] < head[0] else 0.0
        food_right = 1.0 if food[0] > head[0] else 0.0

        dist_now = manhattan(head, food)
        dist_norm = dist_now / (2 * (game.board_size - 1))

        danger_straight = 1.0 if game.is_blocked(straight_cell) is not None else 0.0
        danger_left = 1.0 if game.is_blocked(left_cell) is not None else 0.0
        danger_right = 1.0 if game.is_blocked(right_cell) is not None else 0.0
        danger_back = 1.0 if game.is_blocked(back_cell) is not None else 0.0

        closer_forward = 1.0 if manhattan(straight_cell, food) < dist_now and game.is_blocked(straight_cell) is None else 0.0
        closer_left = 1.0 if manhattan(left_cell, food) < dist_now and game.is_blocked(left_cell) is None else 0.0
        closer_right = 1.0 if manhattan(right_cell, food) < dist_now and game.is_blocked(right_cell) is None else 0.0

        free_neighbors = 0
        for n in self._neighbors(head):
            if game.is_blocked(n) is None:
                free_neighbors += 1
        free_neighbors_norm = free_neighbors / 4.0

        area, food_reachable = self._bfs_area(game, head, allow_tail=True, target=food)
        total_board = game.board_size * game.board_size
        flood_fill_ratio = area / total_board

        # Tail reachability helps reduce getting trapped in loops / dead ends
        tail = game.snake[-1]
        _, tail_reachable = self._bfs_area(game, head, allow_tail=True, target=tail)

        corridor_penalty = 1.0 if free_neighbors <= 1 else 0.0
        length_norm = len(game.snake) / total_board
        head_x_norm = head[0] / (game.board_size - 1)
        head_y_norm = head[1] / (game.board_size - 1)
        near_wall = 1.0 if min(head[0], head[1], game.board_size - 1 - head[0], game.board_size - 1 - head[1]) <= 1 else 0.0
        visited_penalty = min(game.visits[head] / 6.0, 1.0)

        can_eat_next = 1.0 if any(add_pos(head, a) == food for a in game.legal_actions()) else 0.0

        open_straight_3 = self._ray_free_steps(game, d, 3)
        open_left_3 = self._ray_free_steps(game, left, 3)
        open_right_3 = self._ray_free_steps(game, right, 3)

        x = np.array([
            1.0,
            food_up,
            food_down,
            food_left,
            food_right,
            danger_straight,
            danger_left,
            danger_right,
            danger_back,
            dist_norm,
            closer_forward,
            closer_left,
            closer_right,
            free_neighbors_norm,
            flood_fill_ratio,
            1.0 if tail_reachable else 0.0,
            1.0 if food_reachable else 0.0,
            corridor_penalty,
            length_norm,
            head_x_norm,
            head_y_norm,
            near_wall,
            visited_penalty,
            can_eat_next,
            open_straight_3,
            open_left_3,
            open_right_3,
        ], dtype=np.float64)
        return x

    def value(self, game: SnakeGame):
        return float(self.w @ self.features(game))

    def evaluate_action(self, game: SnakeGame, action):
        sim = game.clone()
        result = sim.step(action)
        if result.terminal:
            # terminal target directly
            return result.reward, sim, result
        x_next = self.features(sim)
        return result.reward + self.gamma * float(self.w @ x_next), sim, result

    def choose_action(self, game: SnakeGame, training=True):
        legal = game.legal_actions()
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)

        scored = []
        for a in legal:
            q, _, _ = self.evaluate_action(game, a)
            scored.append((q, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        best_value = scored[0][0]
        best_actions = [a for v, a in scored if abs(v - best_value) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, game: SnakeGame):
        game.reset()
        while not game.terminal:
            x = self.features(game)
            v_hat = float(self.w @ x)

            action = self.choose_action(game, training=True)
            target, next_state, result = self.evaluate_action(game, action)

            # linear TD(0)-style update
            error = target - v_hat
            self.w += self.alpha * error * x

            game = next_state
        return game.score, game.steps, game.death_cause

    def play_episode(self, game: SnakeGame, max_steps=None):
        game.reset()
        if max_steps is None:
            max_steps = 500 + 50 * game.score
        while not game.terminal and game.steps < max_steps:
            action = self.choose_action(game, training=False)
            game.step(action)
        return game.score, game.steps, game.death_cause


def train_agent(
    episodes=8000,
    alpha=0.01,
    gamma=0.95,
    epsilon_start=0.20,
    epsilon_end=0.01,
    seed=0,
    save_path="snake_linear_weights.pkl",
    report_every=100,
):
    rng = random.Random(seed)
    np.random.seed(seed)
    agent = LinearSnakeAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed)
    scores = []

    for ep in range(1, episodes + 1):
        # linear epsilon decay
        frac = (ep - 1) / max(1, episodes - 1)
        agent.epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        game = SnakeGame(seed=rng.randint(0, 10**9))
        score, steps, cause = agent.train_episode(game)
        scores.append(score)

        if ep % report_every == 0:
            recent = scores[-report_every:]
            print(
                f"Episode {ep:5d}/{episodes} | "
                f"avg score={np.mean(recent):.3f} | "
                f"best={np.max(recent)} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    agent.save(save_path)
    print(f"Saved weights to {save_path}")
    return agent, scores


if __name__ == "__main__":
    train_agent()
