"""Pipeline de dados de engenharia: logs de treino → Parquet + DuckDB."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path("data")
EPISODES_TABLE = "episodes"
RUNS_TABLE = "runs"


def new_run_id() -> str:
    """Identificador curto e único para um run de treino."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


class TrainingLogger:
    """Acumula métricas por episódio e persiste em Parquet (particionado) + DuckDB.

    Layout::

        data/
          parquet/episodes/run_id=<id>/part-0.parquet
          parquet/runs/run_id=<id>/part-0.parquet
          duckdb/snake_rl.duckdb
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR, run_id: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.run_id = run_id or new_run_id()
        self.episodes: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        self.started_at = datetime.now(UTC)

        self.parquet_episodes = self.data_dir / "parquet" / "episodes"
        self.parquet_runs = self.data_dir / "parquet" / "runs"
        self.duckdb_path = self.data_dir / "duckdb" / "snake_rl.duckdb"

        self.parquet_episodes.mkdir(parents=True, exist_ok=True)
        self.parquet_runs.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def set_config(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.config["run_id"] = self.run_id
        self.config["started_at"] = self.started_at.isoformat()

    def log_episode(
        self,
        episode: int,
        score: int,
        steps: int,
        total_reward: float,
        epsilon: float,
        loss: float | None,
    ) -> None:
        self.episodes.append(
            {
                "run_id": self.run_id,
                "episode": int(episode),
                "score": int(score),
                "steps": int(steps),
                "total_reward": float(total_reward),
                "epsilon": float(epsilon),
                "loss": float(loss) if loss is not None else None,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def flush(self) -> Path:
        """Escreve Parquet particionado e sincroniza o DuckDB. Retorna o path do DB."""
        if not self.episodes:
            return self.duckdb_path

        ep_df = pd.DataFrame(self.episodes)
        run_row = {
            **self.config,
            "finished_at": datetime.now(UTC).isoformat(),
            "n_episodes": len(self.episodes),
            "best_score": int(ep_df["score"].max()),
            "mean_score": float(ep_df["score"].mean()),
            "final_epsilon": float(ep_df["epsilon"].iloc[-1]),
        }
        # Serializa config aninhado/lista como JSON string para Parquet/DuckDB.
        for key, value in list(run_row.items()):
            if isinstance(value, (dict, list)):
                run_row[key] = json.dumps(value)
        run_df = pd.DataFrame([run_row])

        ep_dir = self.parquet_episodes / f"run_id={self.run_id}"
        run_dir = self.parquet_runs / f"run_id={self.run_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False), ep_dir / "part-0.parquet")
        pq.write_table(
            pa.Table.from_pandas(run_df, preserve_index=False), run_dir / "part-0.parquet"
        )

        self._sync_duckdb()
        return self.duckdb_path

    def _sync_duckdb(self) -> None:
        """Recarrega todas as partições Parquet no arquivo DuckDB."""
        con = duckdb.connect(str(self.duckdb_path))
        try:
            # Hive partitioning: run_id vem do path.
            ep_glob = str(self.parquet_episodes / "**" / "*.parquet")
            run_glob = str(self.parquet_runs / "**" / "*.parquet")

            con.execute(
                f"""
                CREATE OR REPLACE TABLE {EPISODES_TABLE} AS
                SELECT * FROM read_parquet('{ep_glob}', hive_partitioning=true, union_by_name=true)
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {RUNS_TABLE} AS
                SELECT * FROM read_parquet('{run_glob}', hive_partitioning=true, union_by_name=true)
                """
            )
        finally:
            con.close()


def connect_duckdb(data_dir: str | Path = DEFAULT_DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Abre o DuckDB de métricas (somente leitura tipicamente)."""
    path = Path(data_dir) / "duckdb" / "snake_rl.duckdb"
    if not path.exists():
        raise FileNotFoundError(f"DuckDB não encontrado em {path}. Rode `snake train` primeiro.")
    return duckdb.connect(str(path), read_only=True)
