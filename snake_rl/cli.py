"""CLI Typer: snake train | play | gif | report | dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="snake",
    help="Snake RL — Q-learning em NumPy + pipeline Parquet/DuckDB.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def train(
    episodes: Annotated[int, typer.Option("--episodes", "-e", help="Número de episódios")] = 400,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seed global")] = 42,
    grid_size: Annotated[int, typer.Option("--grid-size", help="Tamanho da grade N×N")] = 10,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 0.002,
    gamma: Annotated[float, typer.Option("--gamma", help="Fator de desconto")] = 0.9,
    epsilon_decay: Annotated[float, typer.Option("--epsilon-decay")] = 0.995,
    no_target: Annotated[bool, typer.Option("--no-target", help="Desliga a rede alvo")] = False,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    weights: Annotated[Path, typer.Option("--weights", "-w")] = Path("models/snake_dqn.npz"),
    curve: Annotated[
        Path | None,
        typer.Option("--curve", help="PNG da learning curve (opcional)"),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Treina o agente DQN e grava métricas em Parquet + DuckDB."""
    from snake_rl.render import plot_learning_curve
    from snake_rl.train import TrainConfig
    from snake_rl.train import train as run_train

    cfg = TrainConfig(
        episodes=episodes,
        seed=seed,
        grid_size=grid_size,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        use_target_network=not no_target,
        data_dir=str(data_dir),
        weights_path=str(weights),
    )
    result = run_train(cfg, verbose=not quiet)
    typer.echo(
        f"best={result.best_score} mean={result.mean_score:.2f} "
        f"run_id={result.run_id} duckdb={result.duckdb_path}"
    )
    if curve is not None:
        plot_learning_curve(result.scores, curve)


@app.command()
def play(
    weights: Annotated[Path, typer.Option("--weights", "-w")] = Path("models/snake_dqn.npz"),
    seed: Annotated[int, typer.Option("--seed", "-s")] = 7,
    grid_size: Annotated[int, typer.Option("--grid-size")] = 10,
    render: Annotated[bool, typer.Option("--render/--no-render", help="ASCII no terminal")] = True,
    games: Annotated[int, typer.Option("--games", "-n", help="Partidas de avaliação")] = 1,
) -> None:
    """Joga com pesos salvos (política gulosa)."""
    from snake_rl.render import play_episode
    from snake_rl.train import evaluate

    if not weights.exists():
        typer.echo(f"Pesos não encontrados: {weights}", err=True)
        raise typer.Exit(code=1)

    if games == 1:
        env, _ = play_episode(weights, seed=seed, grid_size=grid_size, render=render)
        typer.echo(f"score final={env.score} steps={env.steps}")
    else:
        stats = evaluate(weights, n_games=games, seed=seed, grid_size=grid_size)
        typer.echo(
            f"games={stats['n_games']} mean={stats['mean_score']:.2f} "
            f"max={stats['max_score']} min={stats['min_score']}"
        )


@app.command()
def gif(
    weights: Annotated[Path, typer.Option("--weights", "-w")] = Path("models/snake_dqn.npz"),
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("docs/snake_play.gif"),
    seed: Annotated[int, typer.Option("--seed", "-s")] = 7,
    grid_size: Annotated[int, typer.Option("--grid-size")] = 10,
    duration: Annotated[int, typer.Option("--duration", help="ms por frame")] = 80,
) -> None:
    """Exporta um GIF do agente treinado."""
    from snake_rl.render import export_gif

    if not weights.exists():
        typer.echo(f"Pesos não encontrados: {weights}", err=True)
        raise typer.Exit(code=1)
    export_gif(weights, out, seed=seed, grid_size=grid_size, duration_ms=duration)


@app.command()
def report(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    window: Annotated[int, typer.Option("--window", help="Janela da média móvel")] = 20,
) -> None:
    """Relatório SQL (média móvel, melhor jogo, curva por run)."""
    from snake_rl.analysis import print_report

    try:
        print_report(data_dir=data_dir, window=window)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def dashboard(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Abre o dashboard Streamlit (requer extra ``dashboard``)."""
    import subprocess
    import sys

    dash = Path(__file__).resolve().parent / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dash),
        "--",
        "--data-dir",
        str(data_dir),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
