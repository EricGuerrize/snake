"""Exporta pesos NumPy e curva de aprendizado para o front-end estático."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from snake_rl.features import FEATURE_NAMES


def weights_to_json(npz_path: str | Path) -> dict[str, Any]:
    """Converte ``.npz`` do agente em um dicionário serializável."""
    data = np.load(npz_path, allow_pickle=False)
    payload: dict[str, Any] = {
        "architecture": "11>64>3",
        "activation": "relu",
        "input_size": int(data["W1"].shape[0]),
        "hidden_size": int(data["W1"].shape[1]),
        "output_size": int(data["W2"].shape[1]),
        "feature_names": FEATURE_NAMES,
        "action_names": ["straight", "right", "left"],
        "W1": data["W1"].astype(np.float64).tolist(),
        "b1": data["b1"].astype(np.float64).tolist(),
        "W2": data["W2"].astype(np.float64).tolist(),
        "b2": data["b2"].astype(np.float64).tolist(),
    }
    if "gamma" in data.files:
        payload["gamma"] = float(data["gamma"])
    if "lr" in data.files:
        payload["lr"] = float(data["lr"])
    return payload


def export_model_json(
    npz_path: str | Path = "models/snake_dqn.npz",
    out_path: str | Path = "web/public/model.json",
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = weights_to_json(npz_path)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out


def export_learning_curve_json(
    data_dir: str | Path = "data",
    out_path: str | Path = "web/public/learning_curve.json",
    window: int = 20,
    run_id: str | None = None,
) -> Path | None:
    """Exporta scores (+ média móvel) do DuckDB para o front-end.

    Retorna ``None`` se não houver dados de treino.
    """
    from snake_rl.analysis import moving_average_score
    from snake_rl.logging_de import connect_duckdb

    try:
        con = connect_duckdb(data_dir)
    except FileNotFoundError:
        return None

    try:
        df = moving_average_score(window=window, run_id=run_id, connection=con)
        if df.empty:
            return None
        if run_id is None:
            # Usa o run mais recente.
            latest = df["run_id"].iloc[-1]
            df = df[df["run_id"] == latest]
        records = [
            {
                "episode": int(r.episode),
                "score": int(r.score),
                "ma_score": float(r.ma_score) if r.ma_score is not None else None,
                "epsilon": float(r.epsilon),
            }
            for r in df.itertuples()
        ]
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": str(df["run_id"].iloc[0]),
            "window": window,
            "episodes": records,
        }
        out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return out
    finally:
        con.close()


def export_parity_fixture(
    npz_path: str | Path = "models/snake_dqn.npz",
    out_path: str | Path = "web/public/parity_fixture.json",
    n_states: int = 16,
    seed: int = 0,
) -> Path:
    """Gera estados de amostra + Q-values Python para o teste de paridade JS."""
    from snake_rl.agent import DQNAgent

    rng = np.random.default_rng(seed)
    agent = DQNAgent(seed=seed)
    agent.load(npz_path)
    agent.epsilon = 0.0

    samples = []
    for _ in range(n_states):
        state = rng.integers(0, 2, size=11).astype(np.float64)
        q = agent.online.forward(state)
        action = int(np.argmax(q))
        samples.append(
            {
                "state": state.tolist(),
                "q": q.astype(np.float64).tolist(),
                "action": action,
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "model.json",
        "atol": 1e-9,
        "samples": samples,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
