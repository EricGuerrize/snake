"""Buffer de experience replay para DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """Uma transição (s, a, r, s', done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fila circular vetorizada em NumPy para amostragem O(batch) de transições."""

    def __init__(self, capacity: int = 50_000, state_dim: int = 11) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self._states = np.empty((capacity, state_dim), dtype=np.float64)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float64)
        self._next_states = np.empty((capacity, state_dim), dtype=np.float64)
        self._dones = np.empty(capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._pos
        self._states[idx] = state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = next_state
        self._dones[idx] = 1.0 if done else 0.0
        self._pos = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Amostra um mini-batch uniformemente usando indexação direta em NumPy."""
        n = self._size
        if batch_size > n:
            raise ValueError(f"batch_size={batch_size} > buffer={n}")
        indices = rng.choice(n, size=batch_size, replace=False)
        return {
            "states": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_states": self._next_states[indices],
            "dones": self._dones[indices],
        }
