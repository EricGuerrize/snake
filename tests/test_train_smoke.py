"""Smoke test de treino curto + pipeline de dados."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
from snake_rl.analysis import best_game, moving_average_score
from snake_rl.train import TrainConfig, evaluate, train


def test_train_few_episodes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    weights = tmp_path / "model.npz"
    cfg = TrainConfig(
        episodes=8,
        seed=0,
        grid_size=8,
        batch_size=16,
        min_replay=16,
        target_update_every=5,
        data_dir=str(data_dir),
        weights_path=str(weights),
        epsilon_decay=0.9,
    )
    result = train(cfg, verbose=False)

    assert weights.exists()
    assert result.duckdb_path.exists()
    assert len(result.scores) == 8
    assert result.best_score >= 0

    # Parquet particionado por run_id.
    ep_parts = list((data_dir / "parquet" / "episodes").rglob("*.parquet"))
    assert ep_parts, "esperado ao menos um part-0.parquet"

    con = duckdb.connect(str(result.duckdb_path), read_only=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert n == 8
        cols = [r[0] for r in con.execute("DESCRIBE episodes").fetchall()]
        for required in (
            "episode",
            "score",
            "steps",
            "total_reward",
            "epsilon",
            "loss",
            "run_id",
            "timestamp",
        ):
            assert required in cols
    finally:
        con.close()

    ma = moving_average_score(window=3, data_dir=data_dir)
    assert len(ma) == 8
    bg = best_game(data_dir=data_dir)
    assert not bg.empty


def test_evaluate_loaded_weights(tmp_path: Path) -> None:
    weights = tmp_path / "m.npz"
    cfg = TrainConfig(
        episodes=3,
        seed=1,
        grid_size=8,
        batch_size=8,
        min_replay=8,
        data_dir=str(tmp_path / "data"),
        weights_path=str(weights),
    )
    train(cfg, verbose=False)
    stats = evaluate(weights, n_games=3, seed=1, grid_size=8)
    assert stats["n_games"] == 3
    assert np.isfinite(stats["mean_score"])
