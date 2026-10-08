### How to run:

1. Install requirements:
```shell
pip install numpy pygame
```
2. Train:
```shell
py train_tictactoe_linear.py --episodes 20000 --out tictactoe_linear_weights.pkl
```
3. Play vs. the trained agent as X:
```shell
py play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl
```
4. AI vs. AI mode:
```shell
py play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl --ai_vs_ai
```

5. Play as O:
```shell
py play_tictactoe_linear.py --weights tictactoe_linear_weights.pkl --human O
```
