"""Dashboard Streamlit: learning curve, distribuição e comparação de runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from snake_rl.analysis import (
    best_game,
    learning_curve_per_run,
    list_runs,
    moving_average_score,
    score_distribution,
)
from snake_rl.logging_de import DEFAULT_DATA_DIR, connect_duckdb


def _parse_data_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args, _ = parser.parse_known_args()
    return Path(args.data_dir)


def main() -> None:
    st.set_page_config(page_title="Snake RL — métricas", layout="wide")
    st.title("Snake RL — dashboard de treino")
    st.caption("Métricas lidas do DuckDB gerado pelo pipeline Parquet.")

    data_dir = _parse_data_dir()
    data_dir = Path(st.sidebar.text_input("data_dir", value=str(data_dir)))
    window = st.sidebar.slider("Janela da média móvel", 5, 50, 20)

    try:
        con = connect_duckdb(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Rode `snake train --episodes 400` para gerar os dados.")
        return

    try:
        runs = list_runs(connection=con)
        if runs.empty:
            st.warning("Tabela de episódios vazia.")
            return

        st.subheader("Runs")
        st.dataframe(runs, use_container_width=True)

        run_ids = runs["run_id"].tolist()
        selected = st.multiselect("Comparar runs", run_ids, default=run_ids[: min(3, len(run_ids))])

        ma = moving_average_score(window=window, connection=con)
        if selected:
            ma = ma[ma["run_id"].isin(selected)]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Learning curve")
            chart = ma.pivot_table(index="episode", columns="run_id", values="ma_score")
            st.line_chart(chart)
        with col2:
            st.subheader("Score bruto")
            raw = ma.pivot_table(index="episode", columns="run_id", values="score")
            st.line_chart(raw)

        st.subheader("Distribuição de scores")
        dist = score_distribution(connection=con)
        if selected:
            dist = dist[dist["run_id"].isin(selected)]
        st.bar_chart(dist.pivot_table(index="score", columns="run_id", values="n", fill_value=0))

        st.subheader("Melhores jogos")
        st.dataframe(best_game(connection=con), use_container_width=True)

        st.subheader("Curva por terço do treino")
        st.dataframe(learning_curve_per_run(connection=con), use_container_width=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
