"""Extração do vetor de estado com 11 features booleanas."""

from __future__ import annotations

import numpy as np

from snake_rl.env import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    Direction,
    SnakeEnv,
)

# Índices do vetor de estado (documentação / testes).
DANGER_STRAIGHT = 0
DANGER_RIGHT = 1
DANGER_LEFT = 2
DIR_RIGHT = 3
DIR_DOWN = 4
DIR_LEFT = 5
DIR_UP = 6
FOOD_LEFT = 7
FOOD_RIGHT = 8
FOOD_UP = 9
FOOD_DOWN = 10

STATE_SIZE = 11
FEATURE_NAMES = [
    "danger_straight",
    "danger_right",
    "danger_left",
    "dir_right",
    "dir_down",
    "dir_left",
    "dir_up",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
]


def extract_state(env: SnakeEnv) -> np.ndarray:
    """Constrói o vetor de 11 bits a partir do ambiente.

    Layout::

        [danger_s, danger_r, danger_l,
         dir_right, dir_down, dir_left, dir_up,
         food_left, food_right, food_up, food_down]
    """
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    head = env.snake[0]
    direction = env.direction

    # Perigo relativo: célula seguinte se seguir reto / virar direita / esquerda.
    for i, action in enumerate((ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT)):
        nxt_dir = SnakeEnv.turn(direction, action)
        nxt_pos = SnakeEnv.next_cell(head, nxt_dir)
        # Para checagem de perigo, consideramos o corpo atual (sem mover a cauda),
        # alinhado à percepção "há obstáculo imediatamente à frente?".
        state[i] = 1.0 if env.is_collision(nxt_pos) else 0.0

    # Direção atual (one-hot).
    state[DIR_RIGHT] = 1.0 if direction == Direction.RIGHT else 0.0
    state[DIR_DOWN] = 1.0 if direction == Direction.DOWN else 0.0
    state[DIR_LEFT] = 1.0 if direction == Direction.LEFT else 0.0
    state[DIR_UP] = 1.0 if direction == Direction.UP else 0.0

    # Posição relativa da comida.
    fx, fy = env.food
    hx, hy = head
    state[FOOD_LEFT] = 1.0 if fx < hx else 0.0
    state[FOOD_RIGHT] = 1.0 if fx > hx else 0.0
    state[FOOD_UP] = 1.0 if fy < hy else 0.0
    state[FOOD_DOWN] = 1.0 if fy > hy else 0.0

    return state
