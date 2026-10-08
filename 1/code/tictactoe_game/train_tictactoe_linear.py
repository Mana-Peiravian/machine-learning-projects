from __future__ import annotations
"""
train_tictactoe_linear_improved.py
"""

import argparse
import os
import numpy as np

from tictactoe_linear_agent import LinearTicTacToeAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon_start", type=float, default=0.20)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.9997)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="tictactoe_linear_weights.pkl")
    parser.add_argument("--eval_games", type=int, default=2000)
    args = parser.parse_args()

    agent = LinearTicTacToeAgent(
        alpha=args.alpha,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        rng_seed=args.seed,
    )

    print("Training started...")
    block = max(1, args.episodes // 20)
    x_wins = 0
    o_wins = 0
    draws = 0

    for ep in range(1, args.episodes + 1):
        info = agent.train_one_episode(random_start=True)
        if info["result"] == 1:
            x_wins += 1
        elif info["result"] == -1:
            o_wins += 1
        else:
            draws += 1

        if ep % block == 0 or ep == args.episodes:
            print(
                f"Episode {ep:6d}/{args.episodes} | "
                f"X wins: {x_wins:5d} | O wins: {o_wins:5d} | Draws: {draws:5d} | "
                f"epsilon={agent.epsilon:.4f}"
            )
            x_wins = o_wins = draws = 0

    agent.save(args.out)
    print(f"\nSaved weights to: {os.path.abspath(args.out)}")

    print("\nLearned weights:")
    print(np.array2string(agent.w, precision=4, suppress_small=False))

    print("\nEvaluation vs random opponent:")
    print("As X ->", agent.play_vs_random(games=args.eval_games, as_player=1))
    print("As O ->", agent.play_vs_random(games=args.eval_games, as_player=-1))

    print("\nRun the graphical version with:")
    print("python play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl")


if __name__ == "__main__":
    main()
