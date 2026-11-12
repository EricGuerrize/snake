"""Ambiente headless do Snake — grid determinístico com seed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Direction(IntEnum):
    """Direção absoluta da cobra na grade."""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Ações relativas à direção atual: 0=seguir reto, 1=virar à direita, 2=virar à esquerda.
ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2

REWARD_FOOD = 10.0
REWARD_CRASH = -10.0
REWARD_STEP = 0.0


@dataclass
class StepResult:
    """Resultado de um passo no ambiente."""

    reward: float
    done: bool
    score: int
    info: dict


class SnakeEnv:
    """Snake clássico em grade N×N, determinístico quando ``seed`` é fixo.

    A cabeça inicia no centro apontando para a direita. A comida é colocada
    em células livres via ``numpy.random.Generator``.
    """

    def __init__(self, grid_size: int = 10, seed: int | None = None) -> None:
        if grid_size < 4:
            raise ValueError("grid_size deve ser >= 4")
        self.grid_size = grid_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.snake: list[tuple[int, int]] = []
        self.direction = Direction.RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.score = 0
        self.steps = 0
        self.done = False
        # Limite de passos sem comer para evitar loops infinitos.
        self.max_steps_without_food = grid_size * grid_size * 2
        self._steps_since_food = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict:
        """Reinicia o episódio. Se ``seed`` for dado, recria o RNG."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        mid = self.grid_size // 2
        # Corpo inicial: 3 células, cabeça à direita.
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        self._steps_since_food = 0
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        if not free:
            # Grade preenchida — vitória técnica; comida fica sob a cabeça.
            self.food = self.snake[0]
            return
        idx = int(self._rng.integers(0, len(free)))
        self.food = free[idx]

    @staticmethod
    def turn(direction: Direction, action: int) -> Direction:
        """Aplica ação relativa e devolve a nova direção absoluta."""
        if action == ACTION_STRAIGHT:
            return direction
        if action == ACTION_RIGHT:
            return Direction((int(direction) + 1) % 4)
        if action == ACTION_LEFT:
            return Direction((int(direction) - 1) % 4)
        raise ValueError(f"ação inválida: {action}")

    @staticmethod
    def next_cell(pos: tuple[int, int], direction: Direction) -> tuple[int, int]:
        x, y = pos
        if direction == Direction.RIGHT:
            return x + 1, y
        if direction == Direction.LEFT:
            return x - 1, y
        if direction == Direction.UP:
            return x, y - 1
        return x, y + 1  # DOWN

    def is_collision(
        self, pos: tuple[int, int], snake: list[tuple[int, int]] | None = None
    ) -> bool:
        """True se a posição está fora da grade ou dentro do corpo."""
        body = snake if snake is not None else self.snake
        x, y = pos
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return True
        return pos in body

    def step(self, action: int) -> StepResult:
        """Executa uma ação relativa e atualiza o estado do jogo."""
        if self.done:
            raise RuntimeError("episódio encerrado — chame reset()")

        self.direction = self.turn(self.direction, action)
        new_head = self.next_cell(self.snake[0], self.direction)
        self.steps += 1
        self._steps_since_food += 1

        # Colisão com parede ou corpo (sem mover a cauda ainda).
        # Ao avançar, a cauda sai — exceto se for comer. Checamos contra o
        # corpo atual menos a cauda, a menos que a comida esteja na nova cabeça.
        will_eat = new_head == self.food
        body_to_check = self.snake if will_eat else self.snake[:-1]
        if self.is_collision(new_head, body_to_check):
            self.done = True
            return StepResult(
                reward=REWARD_CRASH,
                done=True,
                score=self.score,
                info={"reason": "crash"},
            )

        self.snake.insert(0, new_head)
        if will_eat:
            self.score += 1
            self._steps_since_food = 0
            self._place_food()
            reward = REWARD_FOOD
            # Vitória: preencheu a grade.
            if len(self.snake) >= self.grid_size * self.grid_size:
                self.done = True
                return StepResult(
                    reward=reward,
                    done=True,
                    score=self.score,
                    info={"reason": "win"},
                )
        else:
            self.snake.pop()
            reward = REWARD_STEP

        if self._steps_since_food >= self.max_steps_without_food:
            self.done = True
            return StepResult(
                reward=REWARD_CRASH,
                done=True,
                score=self.score,
                info={"reason": "timeout"},
            )

        return StepResult(reward=reward, done=False, score=self.score, info={})

    def get_state(self) -> dict:
        """Snapshot serializável do estado interno (útil para testes/render)."""
        return {
            "snake": list(self.snake),
            "direction": int(self.direction),
            "food": self.food,
            "score": self.score,
            "steps": self.steps,
            "done": self.done,
            "grid_size": self.grid_size,
        }
