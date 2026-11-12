/** Desenha o Snake no canvas (tema escuro / teal com gradiente de corpo e overlay). */

import { DIR, type SnakeEnv } from "./env";

const BG = "#0c0c10";
const GRID = "#181820";
const BORDER = "#323242";
const HEAD = "#e8fffa";
const FOOD = "#f05245";

export type GameRenderExtra = {
  aiSuggestedAction?: number | null;
  humanMode?: boolean;
};

export function drawGame(
  canvas: HTMLCanvasElement,
  env: SnakeEnv,
  extra: GameRenderExtra = {},
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pad = 14;
  const size = Math.min(cssW, cssH) - pad * 2;
  const originX = (cssW - size) / 2;
  const originY = (cssH - size) / 2;
  const cell = size / env.gridSize;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cssW, cssH);

  // Grade sutil
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let i = 1; i < env.gridSize; i++) {
    const p = originX + i * cell;
    ctx.beginPath();
    ctx.moveTo(p, originY);
    ctx.lineTo(p, originY + size);
    ctx.stroke();
    const q = originY + i * cell;
    ctx.beginPath();
    ctx.moveTo(originX, q);
    ctx.lineTo(originX + size, q);
    ctx.stroke();
  }

  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 2;
  ctx.strokeRect(originX, originY, size, size);

  // Se estiver em modo humano, mostra uma dica sutil da célula sugerida pela IA
  if (extra.humanMode && extra.aiSuggestedAction != null && !env.done && env.snake.length > 0) {
    const sugDir = (env.constructor as typeof import("./env").SnakeEnv).turn(
      env.direction,
      extra.aiSuggestedAction,
    );
    const sugCell = (env.constructor as typeof import("./env").SnakeEnv).nextCell(
      env.snake[0],
      sugDir,
    );
    if (
      sugCell.x >= 0 &&
      sugCell.y >= 0 &&
      sugCell.x < env.gridSize &&
      sugCell.y < env.gridSize
    ) {
      const sx = originX + sugCell.x * cell;
      const sy = originY + sugCell.y * cell;
      ctx.save();
      ctx.strokeStyle = "rgba(45, 212, 191, 0.55)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      roundRect(ctx, sx + 4, sy + 4, cell - 8, cell - 8, 5);
      ctx.stroke();
      ctx.restore();
    }
  }

  // Comida com brilho
  const fx = originX + env.food.x * cell + cell / 2;
  const fy = originY + env.food.y * cell + cell / 2;
  const glowGrad = ctx.createRadialGradient(fx, fy, cell * 0.05, fx, fy, cell * 0.48);
  glowGrad.addColorStop(0, "rgba(240, 82, 69, 0.42)");
  glowGrad.addColorStop(1, "rgba(240, 82, 69, 0)");
  ctx.fillStyle = glowGrad;
  ctx.beginPath();
  ctx.arc(fx, fy, cell * 0.48, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = FOOD;
  ctx.beginPath();
  ctx.arc(fx, fy, cell * 0.27, 0, Math.PI * 2);
  ctx.fill();

  // Cobra com degradê cabeça -> cauda para leitura imediata do fluxo do corpo
  const len = Math.max(env.snake.length, 1);
  env.snake.forEach((p, i) => {
    const x = originX + p.x * cell;
    const y = originY + p.y * cell;
    const t = i / Math.max(1, len - 1);
    const inset = i === 0 ? 2 : 2.5 + t * 1.5;
    roundRect(ctx, x + inset, y + inset, cell - inset * 2, cell - inset * 2, i === 0 ? 6 : 5);
    if (i === 0) {
      ctx.fillStyle = HEAD;
    } else {
      const alpha = (1 - t * 0.52).toFixed(2);
      ctx.fillStyle = `rgba(45, 212, 191, ${alpha})`;
    }
    ctx.fill();

    // Olhos na cabeça indicando direção atual
    if (i === 0) {
      drawHeadEyes(ctx, x, y, cell, env.direction);
    }
  });

  // Overlay de fim de partida
  if (env.done) {
    ctx.fillStyle = "rgba(7, 7, 10, 0.68)";
    ctx.fillRect(originX, originY, size, size);

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#e8e8ef";
    ctx.font = "600 18px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.fillText(`Fim de partida · Score ${env.score}`, originX + size / 2, originY + size / 2 - 12);

    ctx.fillStyle = "#8a8a9a";
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.fillText(
      `passos: ${env.steps} · pressione Espaço ou aguarde`,
      originX + size / 2,
      originY + size / 2 + 14,
    );
  }
}

function drawHeadEyes(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cell: number,
  dir: number,
): void {
  const cx = x + cell / 2;
  const cy = y + cell / 2;
  const fwd = cell * 0.16;
  const side = cell * 0.16;
  let e1 = { x: cx, y: cy };
  let e2 = { x: cx, y: cy };

  if (dir === DIR.RIGHT) {
    e1 = { x: cx + fwd, y: cy - side };
    e2 = { x: cx + fwd, y: cy + side };
  } else if (dir === DIR.LEFT) {
    e1 = { x: cx - fwd, y: cy - side };
    e2 = { x: cx - fwd, y: cy + side };
  } else if (dir === DIR.UP) {
    e1 = { x: cx - side, y: cy - fwd };
    e2 = { x: cx + side, y: cy - fwd };
  } else {
    e1 = { x: cx - side, y: cy + fwd };
    e2 = { x: cx + side, y: cy + fwd };
  }

  ctx.fillStyle = "#0a0f12";
  const r = Math.max(1.8, cell * 0.065);
  ctx.beginPath();
  ctx.arc(e1.x, e1.y, r, 0, Math.PI * 2);
  ctx.arc(e2.x, e2.y, r, 0, Math.PI * 2);
  ctx.fill();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
