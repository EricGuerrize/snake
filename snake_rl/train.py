"""Loop de treinamento do agente Snake."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.features import extract_state
from snake_rl.logging_de import TrainingLogger


@dataclass
class TrainConfig:
    episodes: int = 400
    seed: int = 42
    grid_size: int = 10
    gamma: float = 0.9
    lr: float = 0.002
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    hidden_size: int = 64
    use_target_network: bool = True
    target_update_every: int = 100
    min_replay: int = 64
    replay_capacity: int = 50_000
    data_dir: str = "data"
    weights_path: str = "models/snake_dqn.npz"
    run_id: str | None = None


@dataclass
class TrainResult:
    run_id: str
    best_score: int
    mean_score: float
    final_epsilon: float
    weights_path: Path
    duckdb_path: Path
    scores: list[int]


def train(config: TrainConfig | None = None, verbose: bool = True) -> TrainResult:
    """Treina o agente e persiste pesos + logs de engenharia de dados."""
    cfg = config or TrainConfig()
    rng = np.random.default_rng(cfg.seed)

    env = SnakeEnv(grid_size=cfg.grid_size, seed=int(rng.integers(0, 2**31 - 1)))
    agent = DQNAgent(
        seed=cfg.seed,
        hidden_size=cfg.hidden_size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        use_target_network=cfg.use_target_network,
        target_update_every=cfg.target_update_every,
        min_replay=cfg.min_replay,
    )

    logger = TrainingLogger(data_dir=cfg.data_dir, run_id=cfg.run_id)
    logger.set_config({**asdict(cfg), "agent": "DQN-NumPy-MLP"})

    scores: list[int] = []
    best_score = 0
    losses_episode: list[float] = []

    for ep in range(1, cfg.episodes + 1):
        # Seed por episódio para reprodutibilidade do ambiente, sem colapsar a comida.
        ep_seed = int(rng.integers(0, 2**31 - 1))
        env.reset(seed=ep_seed)
        state = extract_state(env)
        total_reward = 0.0
        losses_episode.clear()

        while not env.done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state = extract_state(env)
            agent.remember(state, action, result.reward, next_state, result.done)
            loss = agent.learn()
            if loss is not None:
                losses_episode.append(loss)
            total_reward += result.reward
            state = next_state

        agent.decay_epsilon()
        scores.append(env.score)
        best_score = max(best_score, env.score)
        mean_loss = float(np.mean(losses_episode)) if losses_episode else None

        logger.log_episode(
            episode=ep,
            score=env.score,
            steps=env.steps,
            total_reward=total_reward,
            epsilon=agent.epsilon,
            loss=mean_loss,
        )

        if verbose and (ep % 25 == 0 or ep == 1 or ep == cfg.episodes):
            recent = scores[-25:]
            print(
                f"ep={ep:4d}  score={env.score:3d}  "
                f"best={best_score:3d}  mean25={np.mean(recent):5.2f}  "
                f"eps={agent.epsilon:.3f}  loss={mean_loss if mean_loss is not None else float('nan'):.4f}"
            )

    weights_path = Path(cfg.weights_path)
    agent.save(weights_path)
    duckdb_path = logger.flush()

    result = TrainResult(
        run_id=logger.run_id,
        best_score=best_score,
        mean_score=float(np.mean(scores)),
        final_epsilon=agent.epsilon,
        weights_path=weights_path,
        duckdb_path=duckdb_path,
        scores=scores,
    )
    if verbose:
        print(
            f"\nTreino concluído — run_id={result.run_id}  "
            f"best={result.best_score}  mean={result.mean_score:.2f}  "
            f"weights={weights_path}"
        )
    return result


def evaluate(
    weights_path: str | Path,
    n_games: int = 20,
    seed: int = 0,
    grid_size: int = 10,
    greedy: bool = True,
) -> dict[str, Any]:
    """Avalia um modelo salvo sem exploração."""
    agent = DQNAgent(seed=seed, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(weights_path)
    agent.epsilon = 0.0 if greedy else agent.epsilon

    env = SnakeEnv(grid_size=grid_size, seed=seed)
    scores = []
    for i in range(n_games):
        env.reset(seed=seed + i + 1)
        state = extract_state(env)
        while not env.done:
            action = agent.select_action(state, greedy=greedy)
            env.step(action)
            state = extract_state(env)
        scores.append(env.score)

    return {
        "n_games": n_games,
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "scores": scores,
    }
