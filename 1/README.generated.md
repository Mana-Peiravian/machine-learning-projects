# Homework 1 — Linear Value-Function Learning for Games

## Overview

Feature-based agents learn state values for Snake and Tic-Tac-Toe, with separate training scripts and graphical play interfaces.

## Objectives

The original assignment requests a 20 × 20 Snake environment with a length-three snake and four 2 × 2 obstacle blocks, linear value-function learning, separate training/play programs, and a Tic-Tac-Toe learner using +100/−100/0 terminal targets. A report should explain features and results.

## Implemented approach

Snake implements 27 features, epsilon-greedy exploration, reward shaping, discounting, and TD(0)-style linear updates. Tic-Tac-Toe uses 26 features for position, threats, forks, open lines, and terminal states; self-play evaluates the negative opponent-perspective successor value. Terminal outcomes are also propagated through earlier trajectory states. Both agents save pickle weights and provide Pygame interfaces.

## Repository structure

| Path | Purpose |
| --- | --- |
| [Homework1.pdf](Homework1.pdf) | Original assignment |
| [HW1-G.pdf](HW1-G.pdf) | Supplementary guide |
| [HW#1 - Report.pdf](HW%231%20-%20Report.pdf) | Submitted report |
| [HW#1 - Report.docx](HW%231%20-%20Report.docx) | Editable report |
| [code/snake_game/](code/snake_game/) | Environment, agent, train/play scripts, saved weights |
| [code/tictactoe_game/](code/tictactoe_game/) | Agent, train/play scripts, saved weights |
| [README.md](README.md) | Existing documentation, preserved |

## Requirements

Python, NumPy and Pygame; versions are not pinned. A graphical desktop is needed for play. Load saved pickle weights only if their provenance is trusted.

## Input data

Randomly generated game states; no external dataset. Two saved weight pickles are available for play.

## Running the project

Commands are inferred from argument parsers and existing game READMEs; **not tested**. Install NumPy and Pygame in a separate environment. From the repository root:

```bash
cd 1/code/snake_game
python play_snake_linear.py --weights snake_linear_weights.pkl
```

For Tic-Tac-Toe, start again from the repository root:

```bash
cd 1/code/tictactoe_game
python play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl
```

Snake supports --manual; Tic-Tac-Toe supports --ai_vs_ai and --human O. These open interactive windows. Optional training writes weights; use a disposable copy and new output names:

```bash
# From Snake directory:
python train_snake_linear.py --episodes 8000 --out new_snake_weights.pkl
# From Tic-Tac-Toe directory:
python train_tictactoe_linear.py --episodes 20000 --out new_tictactoe_weights.pkl
```

## Results

The report describes better food seeking and hazard avoidance in Snake, with moderate scores and looping limitations. It claims stronger center control, blocking, and rare losses after Tic-Tac-Toe training. These are qualitative report claims, not independently measured results. Saved weights are present; no evaluation logs or numeric performance tables are preserved.

## Assignment materials

- [Submitted report](HW%231%20-%20Report.pdf)
- [Implementation](code/)
- [Original assignment](Homework1.pdf)
- [Supplementary guide](HW1-G.pdf)

## Notes and limitations

Code adds shaping/discounting beyond the simplified assignment description. Static inspection shows Snake’s _bfs_area is called on the occupied head and returns immediately through is_blocked(start); area/reachability features at those call sites return zero/false. No behavior test or code correction was performed. The existing README links an absent submission ZIP. The report PDF/DOCX contain student identifiers; review before release.

## Academic context

Completed as part of a master's-level Machine Learning course; documented for portfolio and educational use. See the [root README](../README.md) for attribution and [license notice](../LICENSE-NOTICE.md) for ownership.
