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
    """Fila circular de transições amostradas uniformemente."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append(
            Transition(
                state=np.asarray(state, dtype=np.float64),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float64),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Amostra um mini-batch e empilha em arrays NumPy."""
        n = len(self._buf)
        if batch_size > n:
            raise ValueError(f"batch_size={batch_size} > buffer={n}")
        indices = rng.choice(n, size=batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        return {
            "states": np.stack([t.state for t in batch]),
            "actions": np.array([t.action for t in batch], dtype=np.int64),
            "rewards": np.array([t.reward for t in batch], dtype=np.float64),
            "next_states": np.stack([t.next_state for t in batch]),
            "dones": np.array([t.done for t in batch], dtype=np.float64),
        }
