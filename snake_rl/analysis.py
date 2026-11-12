"""Análises SQL sobre os logs de treino no DuckDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from snake_rl.logging_de import DEFAULT_DATA_DIR, connect_duckdb


def _con(data_dir: str | Path | None, connection: duckdb.DuckDBPyConnection | None):
    if connection is not None:
        return connection, False
    return connect_duckdb(data_dir or DEFAULT_DATA_DIR), True


def moving_average_score(
    window: int = 20,
    run_id: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Média móvel do score por episódio (learning curve suavizada)."""
    con, owns = _con(data_dir, connection)
    try:
        where = "WHERE run_id = ?" if run_id else ""
        params: list[Any] = [run_id] if run_id else []
        # DuckDB window functions.
        sql = f"""
            SELECT
                run_id,
                episode,
                score,
                AVG(score) OVER (
                    PARTITION BY run_id
                    ORDER BY episode
                    ROWS BETWEEN {int(window) - 1} PRECEDING AND CURRENT ROW
                ) AS ma_score,
                total_reward,
                epsilon,
                loss
            FROM episodes
            {where}
            ORDER BY run_id, episode
        """
        return con.execute(sql, params).df()
    finally:
        if owns:
            con.close()


def best_game(
    run_id: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Melhor episódio (maior score) por run, ou globalmente."""
    con, owns = _con(data_dir, connection)
    try:
        where = "WHERE run_id = ?" if run_id else ""
        params: list[Any] = [run_id] if run_id else []
        sql = f"""
            SELECT run_id, episode, score, steps, total_reward, epsilon, loss, timestamp
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY score DESC, episode ASC) AS rn
                FROM episodes
                {where}
            )
            WHERE rn = 1
            ORDER BY score DESC
        """
        return con.execute(sql, params).df()
    finally:
        if owns:
            con.close()


def learning_curve_per_run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Resumo da curva de aprendizado por run (início / meio / fim)."""
    con, owns = _con(data_dir, connection)
    try:
        sql = """
            WITH ranked AS (
                SELECT
                    run_id,
                    episode,
                    score,
                    NTILE(3) OVER (PARTITION BY run_id ORDER BY episode) AS tertile
                FROM episodes
            )
            SELECT
                run_id,
                tertile,
                COUNT(*) AS n_episodes,
                AVG(score) AS mean_score,
                MAX(score) AS max_score,
                MIN(score) AS min_score
            FROM ranked
            GROUP BY run_id, tertile
            ORDER BY run_id, tertile
        """
        return con.execute(sql).df()
    finally:
        if owns:
            con.close()


def score_distribution(
    run_id: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Histograma simples de scores."""
    con, owns = _con(data_dir, connection)
    try:
        where = "WHERE run_id = ?" if run_id else ""
        params: list[Any] = [run_id] if run_id else []
        sql = f"""
            SELECT run_id, score, COUNT(*) AS n
            FROM episodes
            {where}
            GROUP BY run_id, score
            ORDER BY run_id, score
        """
        return con.execute(sql, params).df()
    finally:
        if owns:
            con.close()


def list_runs(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Lista runs disponíveis com métricas agregadas."""
    con, owns = _con(data_dir, connection)
    try:
        # A tabela runs pode ter colunas variáveis; pegamos o essencial via episodes.
        sql = """
            SELECT
                run_id,
                COUNT(*) AS n_episodes,
                MAX(score) AS best_score,
                AVG(score) AS mean_score,
                MIN(timestamp) AS started,
                MAX(timestamp) AS finished
            FROM episodes
            GROUP BY run_id
            ORDER BY started DESC
        """
        return con.execute(sql).df()
    finally:
        if owns:
            con.close()


def print_report(data_dir: str | Path = DEFAULT_DATA_DIR, window: int = 20) -> None:
    """Imprime um relatório textual no terminal."""
    con = connect_duckdb(data_dir)
    try:
        runs = list_runs(connection=con)
        if runs.empty:
            print("Nenhum episódio encontrado no DuckDB.")
            return

        print("=== Runs ===")
        print(runs.to_string(index=False))
        print()

        print("=== Melhores jogos ===")
        print(best_game(connection=con).to_string(index=False))
        print()

        print(f"=== Média móvel (janela={window}) — últimos 5 por run ===")
        ma = moving_average_score(window=window, connection=con)
        for rid, grp in ma.groupby("run_id"):
            print(f"\nrun_id={rid}")
            print(grp.tail(5)[["episode", "score", "ma_score", "epsilon"]].to_string(index=False))

        print("\n=== Curva por terço do treino ===")
        print(learning_curve_per_run(connection=con).to_string(index=False))
    finally:
        con.close()
