"""Agente DQN com MLP implementado em NumPy puro (sem torch/tensorflow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from snake_rl.features import STATE_SIZE
from snake_rl.replay import ReplayBuffer

N_ACTIONS = 3
HIDDEN_SIZE = 64


class MLPQNetwork:
    """MLP 11 → 64 (ReLU) → 3 com forward e backprop manuais.

    Forward::

        h = max(0, x @ W1 + b1)
        q = h @ W2 + b2

    Gradientes (MSE em cima do Q da ação tomada)::

        ∂L/∂Q = 2 (Q − y)   # apenas na ação a; demais saídas zeradas
        dW2 = hᵀ @ dq ,  db2 = sum(dq)
        dh  = (dq @ W2ᵀ) ⊙ (h > 0)
        dW1 = xᵀ @ dh ,  db1 = sum(dh)
    """

    def __init__(
        self,
        input_size: int = STATE_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = N_ACTIONS,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = rng or np.random.default_rng()

        # Inicialização He / Xavier leve para estabilidade.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size, dtype=np.float64)

        # Cache do último forward (usado no backward).
        self._cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Passagem para frente. ``x`` shape (batch, input) ou (input,)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        x = np.asarray(x, dtype=np.float64)
        z1 = x @ self.W1 + self.b1
        h = np.maximum(0.0, z1)  # ReLU
        q = h @ self.W2 + self.b2
        self._cache = {"x": x, "h": h, "q": q}
        return q[0] if single else q

    def backward(
        self,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Backprop MSE só nas ações tomadas; atualiza pesos in-place.

        Parameters
        ----------
        actions:
            Índices das ações no batch, shape (batch,).
        targets:
            Alvos TD ``y = r + γ max Q(s',·)``, shape (batch,).
        learning_rate:
            Passo do SGD.

        Returns
        -------
        float
            Loss MSE média do batch.
        """
        x = self._cache["x"]
        h = self._cache["h"]
        q = self._cache["q"]
        batch = q.shape[0]

        # Predições Q(s, a) e residual.
        q_sa = q[np.arange(batch), actions]
        residual = q_sa - targets  # (batch,)
        loss = float(np.mean(residual**2))

        # ∂L/∂Q: 2(Q−y)/batch apenas na coluna da ação.
        dq = np.zeros_like(q)
        dq[np.arange(batch), actions] = (2.0 / batch) * residual

        # Camada de saída.
        dW2 = h.T @ dq
        db2 = dq.sum(axis=0)

        # ReLU backward.
        dh = (dq @ self.W2.T) * (h > 0)

        dW1 = x.T @ dh
        db1 = dh.sum(axis=0)

        # SGD.
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return loss

    def copy_weights_from(self, other: MLPQNetwork) -> None:
        """Copia profunda dos parâmetros (rede alvo)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def get_weights(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.W1 = np.asarray(weights["W1"], dtype=np.float64)
        self.b1 = np.asarray(weights["b1"], dtype=np.float64)
        self.W2 = np.asarray(weights["W2"], dtype=np.float64)
        self.b2 = np.asarray(weights["b2"], dtype=np.float64)

    def numerical_gradient_check(
        self,
        x: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        eps: float = 1e-5,
    ) -> dict[str, float]:
        """Compara ∂L/∂θ analítico com diferenças finitas.

        Não atualiza os pesos — apenas mede o erro relativo máximo.
        """
        # Forward + gradientes analíticos sem SGD.
        self.forward(x if x.ndim == 2 else x.reshape(1, -1))
        xc = self._cache["x"]
        h = self._cache["h"]
        q = self._cache["q"]
        batch = q.shape[0]
        q_sa = q[np.arange(batch), actions]
        residual = q_sa - targets
        dq = np.zeros_like(q)
        dq[np.arange(batch), actions] = (2.0 / batch) * residual

        analytic = {
            "W2": h.T @ dq,
            "b2": dq.sum(axis=0),
            "W1": xc.T @ ((dq @ self.W2.T) * (h > 0)),
            "b1": ((dq @ self.W2.T) * (h > 0)).sum(axis=0),
        }

        def loss_at() -> float:
            qq = self.forward(xc)
            return float(np.mean((qq[np.arange(batch), actions] - targets) ** 2))

        rel_errors: dict[str, float] = {}
        for name, param in (
            ("W1", self.W1),
            ("b1", self.b1),
            ("W2", self.W2),
            ("b2", self.b2),
        ):
            numeric = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                old = param[idx]
                param[idx] = old + eps
                loss_plus = loss_at()
                param[idx] = old - eps
                loss_minus = loss_at()
                param[idx] = old
                numeric[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                it.iternext()

            a = analytic[name]
            denom = np.maximum(np.abs(a) + np.abs(numeric), 1e-8)
            rel_errors[name] = float(np.max(np.abs(a - numeric) / denom))

        # Restaura cache limpo.
        self.forward(xc)
        return rel_errors


class DQNAgent:
    """Agente ε-greedy com MLP NumPy, replay e rede alvo opcional."""

    def __init__(
        self,
        seed: int | None = 42,
        hidden_size: int = HIDDEN_SIZE,
        gamma: float = 0.9,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        use_target_network: bool = True,
        target_update_every: int = 50,
        min_replay: int = 200,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_target_network = use_target_network
        self.target_update_every = target_update_every
        self.min_replay = min_replay

        self.online = MLPQNetwork(hidden_size=hidden_size, rng=self.rng)
        self.target = MLPQNetwork(hidden_size=hidden_size, rng=self.rng)
        self.target.copy_weights_from(self.online)

        self.replay = ReplayBuffer(capacity=replay_capacity)
        self.train_steps = 0
        self.last_loss: float | None = None

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Política ε-greedy sobre os Q-valores."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        q = self.online.forward(state)
        return int(np.argmax(q))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """Amostra o replay e dá um passo de SGD. Retorna a loss ou None."""
        if len(self.replay) < max(self.min_replay, self.batch_size):
            return None

        batch = self.replay.sample(self.batch_size, self.rng)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Alvo TD: y = r + γ max_a' Q_target(s', a')  (ou 0 se terminal).
        net = self.target if self.use_target_network else self.online
        next_q = net.forward(next_states)
        max_next = next_q.max(axis=1)
        targets = rewards + self.gamma * max_next * (1.0 - dones)

        self.online.forward(states)
        loss = self.online.backward(actions, targets, self.lr)
        self.last_loss = loss
        self.train_steps += 1

        if self.use_target_network and self.train_steps % self.target_update_every == 0:
            self.target.copy_weights_from(self.online)

        return loss

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Salva pesos e hiperparâmetros em ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "W1": self.online.W1,
            "b1": self.online.b1,
            "W2": self.online.W2,
            "b2": self.online.b2,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "lr": self.lr,
            "train_steps": self.train_steps,
            "hidden_size": self.online.hidden_size,
            "use_target_network": self.use_target_network,
        }
        np.savez(path, **payload)

    def load(self, path: str | Path) -> None:
        """Carrega pesos de um ``.npz`` salvo por :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        self.online.set_weights(
            {"W1": data["W1"], "b1": data["b1"], "W2": data["W2"], "b2": data["b2"]}
        )
        self.target.copy_weights_from(self.online)
        self.epsilon = float(data["epsilon"]) if "epsilon" in data.files else self.epsilon_end
        if "gamma" in data.files:
            self.gamma = float(data["gamma"])
        if "lr" in data.files:
            self.lr = float(data["lr"])
        if "train_steps" in data.files:
            self.train_steps = int(data["train_steps"])
