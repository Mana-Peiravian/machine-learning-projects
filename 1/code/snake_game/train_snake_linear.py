import argparse
from snake_linear_agent import train_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_start", type=float, default=0.20)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="./snake_linear_weights.pkl")
    args = parser.parse_args()

    train_agent(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
        save_path=args.out,
    )

if __name__ == "__main__":
    main()