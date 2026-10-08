### How to run:


1. Install requirements:
```shell
pip install numpy pygame
```
2. Train:
```shell
py train_snake_linear.py --episodes 8000 --out snake_linear_weights.pkl
```
3. Play with the trained agent:
```shell
py play_snake_linear.py --weights snake_linear_weights.pkl
```
4. Or play manually:
```shell
py play_snake_linear.py --manual
```