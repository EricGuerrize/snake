"""Testes das regras do ambiente Snake."""

from __future__ import annotations

import pytest
from snake_rl.env import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    REWARD_CRASH,
    REWARD_FOOD,
    Direction,
    SnakeEnv,
)


def test_reset_deterministic_with_seed() -> None:
    a = SnakeEnv(grid_size=10, seed=123)
    b = SnakeEnv(grid_size=10, seed=123)
    assert a.snake == b.snake
    assert a.food == b.food
    assert a.direction == b.direction


def test_different_seeds_differ() -> None:
    a = SnakeEnv(grid_size=10, seed=1)
    b = SnakeEnv(grid_size=10, seed=2)
    # Comida deve diferir com alta probabilidade em grade 10x10.
    assert a.food != b.food or a.snake != b.snake


def test_turn_relative_actions() -> None:
    assert SnakeEnv.turn(Direction.RIGHT, ACTION_STRAIGHT) == Direction.RIGHT
    assert SnakeEnv.turn(Direction.RIGHT, ACTION_RIGHT) == Direction.DOWN
    assert SnakeEnv.turn(Direction.RIGHT, ACTION_LEFT) == Direction.UP
    assert SnakeEnv.turn(Direction.UP, ACTION_RIGHT) == Direction.RIGHT
    assert SnakeEnv.turn(Direction.UP, ACTION_LEFT) == Direction.LEFT


def test_wall_collision_gives_crash_reward() -> None:
    env = SnakeEnv(grid_size=8, seed=0)
    # Empurra a cobra contra a parede direita.
    env.snake = [(7, 4), (6, 4), (5, 4)]
    env.direction = Direction.RIGHT
    env.food = (0, 0)
    result = env.step(ACTION_STRAIGHT)
    assert result.done
    assert result.reward == REWARD_CRASH
    assert result.info["reason"] == "crash"


def test_self_collision() -> None:
    env = SnakeEnv(grid_size=10, seed=0)
    # Corpo em U: ao ir reto a cabeça bate no próprio corpo.
    env.snake = [(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)]
    env.direction = Direction.LEFT
    env.food = (0, 0)
    # Virar à direita (LEFT dir → UP) não; vamos forçar ação que causa colisão.
    # Direção LEFT, ACTION_LEFT → DOWN → (5,6) que está no corpo.
    result = env.step(ACTION_LEFT)
    assert result.done
    assert result.reward == REWARD_CRASH


def test_eat_food_grows_and_scores() -> None:
    env = SnakeEnv(grid_size=10, seed=0)
    env.snake = [(3, 5), (2, 5), (1, 5)]
    env.direction = Direction.RIGHT
    env.food = (4, 5)
    length_before = len(env.snake)
    result = env.step(ACTION_STRAIGHT)
    assert result.reward == REWARD_FOOD
    assert env.score == 1
    assert len(env.snake) == length_before + 1
    assert env.food != (4, 5)


def test_step_after_done_raises() -> None:
    env = SnakeEnv(grid_size=6, seed=0)
    env.snake = [(5, 3), (4, 3), (3, 3)]
    env.direction = Direction.RIGHT
    env.food = (0, 0)
    env.step(ACTION_STRAIGHT)
    assert env.done
    with pytest.raises(RuntimeError):
        env.step(ACTION_STRAIGHT)


def test_tail_vacates_cell() -> None:
    """Avançar sem comer libera a célula da cauda — não deve colidir."""
    env = SnakeEnv(grid_size=10, seed=0)
    env.snake = [(4, 4), (3, 4), (2, 4)]
    env.direction = Direction.RIGHT
    env.food = (9, 9)
    result = env.step(ACTION_STRAIGHT)
    assert not result.done
    assert env.snake[0] == (5, 4)
    assert (2, 4) not in env.snake


def test_grid_size_validation() -> None:
    with pytest.raises(ValueError):
        SnakeEnv(grid_size=3)
