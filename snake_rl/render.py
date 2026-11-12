"""Renderização no terminal e exportação de GIF / gráficos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.features import extract_state

# Cores no estilo do post de inspiração: grade escura, cobra verde, comida vermelha.
COLOR_BG = (18, 18, 24)
COLOR_GRID = (36, 36, 48)
COLOR_SNAKE = (46, 204, 113)
COLOR_HEAD = (39, 174, 96)
COLOR_FOOD = (231, 76, 60)
COLOR_TEXT = (220, 220, 230)


def frame_to_ascii(env: SnakeEnv) -> str:
    """Representação ASCII da grade (para ``snake play --render``)."""
    grid = [["·"] * env.grid_size for _ in range(env.grid_size)]
    fx, fy = env.food
    grid[fy][fx] = "●"
    for i, (x, y) in enumerate(env.snake):
        grid[y][x] = "█" if i == 0 else "○"
    lines = [" ".join(row) for row in grid]
    header = f"score={env.score}  steps={env.steps}  dir={env.direction.name}"
    return header + "\n" + "\n".join(lines)


def _draw_frame(env: SnakeEnv, cell: int = 20, pad: int = 4) -> Image.Image:
    size = env.grid_size * cell + pad * 2
    img = Image.new("RGB", (size, size + 28), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Grade.
    for i in range(env.grid_size + 1):
        x0 = pad + i * cell
        draw.line([(x0, pad), (x0, pad + env.grid_size * cell)], fill=COLOR_GRID)
        draw.line(
            [(pad, pad + i * cell), (pad + env.grid_size * cell, pad + i * cell)], fill=COLOR_GRID
        )

    # Comida.
    fx, fy = env.food
    draw.ellipse(
        [
            pad + fx * cell + 3,
            pad + fy * cell + 3,
            pad + (fx + 1) * cell - 3,
            pad + (fy + 1) * cell - 3,
        ],
        fill=COLOR_FOOD,
    )

    # Corpo.
    for i, (x, y) in enumerate(env.snake):
        color = COLOR_HEAD if i == 0 else COLOR_SNAKE
        draw.rounded_rectangle(
            [
                pad + x * cell + 2,
                pad + y * cell + 2,
                pad + (x + 1) * cell - 2,
                pad + (y + 1) * cell - 2,
            ],
            radius=4,
            fill=color,
        )

    draw.text((pad, size + 6), f"score={env.score}  steps={env.steps}", fill=COLOR_TEXT)
    return img


def play_episode(
    weights_path: str | Path,
    seed: int = 0,
    grid_size: int = 10,
    render: bool = False,
    max_steps: int = 2000,
) -> tuple[SnakeEnv, list[Image.Image]]:
    """Joga um episódio guloso e opcionalmente coleta frames / imprime ASCII."""
    agent = DQNAgent(seed=seed, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(weights_path)
    agent.epsilon = 0.0

    env = SnakeEnv(grid_size=grid_size, seed=seed)
    env.reset(seed=seed)
    state = extract_state(env)
    frames: list[Image.Image] = [_draw_frame(env)]

    if render:
        print(frame_to_ascii(env))
        print()

    steps = 0
    while not env.done and steps < max_steps:
        action = agent.select_action(state, greedy=True)
        env.step(action)
        state = extract_state(env)
        frames.append(_draw_frame(env))
        steps += 1
        if render:
            print(frame_to_ascii(env))
            print()

    return env, frames


def export_gif(
    weights_path: str | Path,
    out_path: str | Path,
    seed: int = 7,
    grid_size: int = 10,
    duration_ms: int = 80,
    max_steps: int = 800,
) -> Path:
    """Exporta um GIF do agente treinado jogando."""
    env, frames = play_episode(
        weights_path=weights_path,
        seed=seed,
        grid_size=grid_size,
        render=False,
        max_steps=max_steps,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("nenhum frame gerado")
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"GIF salvo em {out} (score={env.score}, frames={len(frames)})")
    return out


def plot_learning_curve(
    scores: list[int],
    out_path: str | Path,
    window: int = 20,
    title: str = "score per game while training",
) -> Path:
    """Gera o PNG da curva de aprendizado (estilo do post de inspiração)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores_arr = np.asarray(scores, dtype=np.float64)
    episodes = np.arange(1, len(scores_arr) + 1)

    if len(scores_arr) >= window:
        kernel = np.ones(window) / window
        ma = np.convolve(scores_arr, kernel, mode="valid")
        ma_x = episodes[window - 1 :]
    else:
        ma = scores_arr
        ma_x = episodes

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.set_facecolor("#111118")
    fig.patch.set_facecolor("#0b0b10")
    ax.scatter(episodes, scores_arr, s=8, c="#5dade2", alpha=0.45, label="score")
    ax.plot(ma_x, ma, color="#f5b041", linewidth=2.0, label=f"média móvel ({window})")
    ax.set_xlabel("game number", color="#ddd")
    ax.set_ylabel("apples eaten", color="#ddd")
    ax.set_title(title, color="#eee", fontsize=13, pad=10)
    ax.tick_params(colors="#bbb")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(facecolor="#1a1a22", edgecolor="#333", labelcolor="#ddd")
    ax.grid(True, alpha=0.15, color="#888")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Curva salva em {out}")
    return out
