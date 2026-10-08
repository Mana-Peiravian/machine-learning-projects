# Homework 1 — Linear Value-Function Learning for Games

## Overview

This assignment studies feature-based linear value approximation in two environments: a 20 × 20 Snake game and Tic-Tac-Toe. The submitted implementation learns weights from interaction rather than storing a value for every state, and includes graphical play programs for both games.

## Objectives

The assignment asks for:

- A Snake agent on a 20 × 20 board with a length-three snake, four random 2 × 2 obstacle blocks, and one food item.
- Greedy successor-state selection using a learned linear value function.
- Terminal penalties of −1000 for losing Snake states.
- A Tic-Tac-Toe agent using the same approximation idea, with +100/−100/0 terminal values for win/loss/draw.
- Separate training and interactive play programs, plus a report explaining features and results.

## Implemented approach

### Snake

`LinearSnakeAgent` uses 27 engineered features. They describe food direction and distance, immediate danger, local openness, short free-space rays, flood-fill area, tail and food reachability, corridor risk, revisits, position, and snake length. Action values combine the immediate reward with a discounted estimate of the successor state. Training uses a TD(0)-style linear update and decaying epsilon-greedy exploration.

The environment implements random valid starts, obstacle placement, collision handling, food rewards, revisit penalties, a small distance-based shaping reward, and timeout termination. `play_snake_linear.py` provides manual and trained-agent Pygame modes.

### Tic-Tac-Toe

The board is encoded from the current player's perspective. The 26-feature representation includes center/corner/edge control, open lines, immediate wins, threats, forks, and terminal indicators. Training uses self-play, random legal starting states, epsilon decay, and the opponent-perspective target `-V(next_state)`. A Pygame interface supports human-vs-agent and agent-vs-agent play.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`Homework1.pdf`](Homework1.pdf) | Instructor assignment description |
| [`HW1-G.pdf`](HW1-G.pdf) | Supplementary assignment guide |
| [`HW#1 - Report.pdf`](HW%231%20-%20Report.pdf) | Submitted report |
| [`code/snake_game/`](code/snake_game/) | Snake environment, learner, training, GUI, and saved weights |
| [`code/tictactoe_game/`](code/tictactoe_game/) | Tic-Tac-Toe learner, training, GUI, and saved weights |
| [`ManaPeiravian_SinaSabooki.zip`](ManaPeiravian_SinaSabooki.zip) | Original submission archive |

## Requirements

- Python
- NumPy
- Pygame

No package versions are pinned.

## Running the project

These commands are confirmed from the source READMEs and argument parsers but were not executed during portfolio generation.

### Snake

```bash
cd 1/code/snake_game
python -m pip install numpy pygame
python train_snake_linear.py --episodes 8000 --out snake_linear_weights.pkl
python play_snake_linear.py --weights snake_linear_weights.pkl
```

Manual mode:

```bash
python play_snake_linear.py --manual
```

### Tic-Tac-Toe

```bash
cd 1/code/tictactoe_game
python -m pip install numpy pygame
python train_tictactoe_linear.py --episodes 20000 --out tictactoe_linear_weights.pkl
python play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl
```

## Results

The report qualitatively states that the trained Snake agent moves toward food, avoids immediate hazards, and reduces self-collisions, while remaining limited by loops and the linear model. It describes the Tic-Tac-Toe agent as learning center control, blocking, and winning opportunities and as rarely losing after sufficient training.

These are report claims rather than independently reproduced measurements. No numeric evaluation table or training log is preserved in the visible folder. Saved weight files provide evidence of trained artifacts, but they were not loaded or evaluated during this audit.

## Notes and limitations

- The training workflows are stochastic and may be time-consuming.
- There is no pinned environment or automated test suite.
- Performance claims are qualitative and unverified.
- Python pickle files should only be loaded when their provenance is trusted.
- The report contains personal student identifiers and should be redacted before public release.

## Academic context

> This project was completed as part of a master's-level Machine Learning course. It is documented for portfolio and educational purposes.

The submitted report names Mana Peiravian and Sina Sabooki as collaborators.

