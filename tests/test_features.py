"""Testes do vetor de estado com 11 features."""

from __future__ import annotations

import numpy as np
from snake_rl.env import Direction, SnakeEnv
from snake_rl.features import (
    DANGER_LEFT,
    DANGER_RIGHT,
    DANGER_STRAIGHT,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    FOOD_DOWN,
    FOOD_LEFT,
    FOOD_RIGHT,
    FOOD_UP,
    STATE_SIZE,
    extract_state,
)


def test_state_shape_and_binary() -> None:
    env = SnakeEnv(grid_size=10, seed=42)
    state = extract_state(env)
    assert state.shape == (STATE_SIZE,)
    assert state.dtype == np.float64
    assert set(np.unique(state)).issubset({0.0, 1.0})


def test_direction_one_hot() -> None:
    env = SnakeEnv(grid_size=10, seed=0)
    for direction, idx in (
        (Direction.RIGHT, DIR_RIGHT),
        (Direction.DOWN, DIR_DOWN),
        (Direction.LEFT, DIR_LEFT),
        (Direction.UP, DIR_UP),
    ):
        env.direction = direction
        state = extract_state(env)
        one_hots = [state[DIR_RIGHT], state[DIR_DOWN], state[DIR_LEFT], state[DIR_UP]]
        assert sum(one_hots) == 1.0
        assert state[idx] == 1.0


def test_danger_near_wall() -> None:
    env = SnakeEnv(grid_size=8, seed=0)
    # Cabeça colada na parede direita, indo para a direita → perigo reto.
    env.snake = [(7, 4), (6, 4), (5, 4)]
    env.direction = Direction.RIGHT
    env.food = (0, 0)
    state = extract_state(env)
    assert state[DANGER_STRAIGHT] == 1.0
    # Virar à direita → DOWN (7,5) livre; esquerda → UP (7,3) livre.
    assert state[DANGER_RIGHT] == 0.0
    assert state[DANGER_LEFT] == 0.0


def test_food_relative_flags() -> None:
    env = SnakeEnv(grid_size=12, seed=0)
    env.snake = [(5, 5), (4, 5), (3, 5)]
    env.direction = Direction.RIGHT
    env.food = (8, 2)  # à direita e acima
    state = extract_state(env)
    assert state[FOOD_RIGHT] == 1.0
    assert state[FOOD_LEFT] == 0.0
    assert state[FOOD_UP] == 1.0
    assert state[FOOD_DOWN] == 0.0


def test_food_same_cell_flags_zero() -> None:
    """Se comida coincidisse com a cabeça (edge), nenhum flag relativo."""
    env = SnakeEnv(grid_size=10, seed=0)
    env.snake = [(5, 5), (4, 5), (3, 5)]
    env.food = (5, 5)
    state = extract_state(env)
    assert state[FOOD_LEFT] == 0.0
    assert state[FOOD_RIGHT] == 0.0
    assert state[FOOD_UP] == 0.0
    assert state[FOOD_DOWN] == 0.0


def test_tail_tip_not_flagged_as_danger_when_not_eating() -> None:
    """A ponta da cauda sai da célula ao avançar sem comer; não deve ser marcada como perigo."""
    env = SnakeEnv(grid_size=10, seed=0)
    # Quadrado 2x2: cabeça em (5,5) indo para UP -> célula acima (5,4) é a ponta da cauda.
    env.snake = [(5, 5), (4, 5), (4, 4), (5, 4)]
    env.direction = Direction.UP
    env.food = (0, 0)
    state = extract_state(env)
    assert state[DANGER_STRAIGHT] == 0.0
