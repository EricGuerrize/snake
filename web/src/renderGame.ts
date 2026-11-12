/** Desenha o Snake no canvas (tema escuro / teal). */

import type { SnakeEnv } from "./env";

const BG = "#0c0c10";
const GRID = "#1c1c24";
const BORDER = "#3a3a48";
const SNAKE = "#2dd4bf";
const HEAD = "#e8fffa";
const FOOD = "#e74c3c";

export function drawGame(canvas: HTMLCanvasElement, env: SnakeEnv): void {
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

  const pad = 10;
  const size = Math.min(cssW, cssH) - pad * 2;
  const originX = (cssW - size) / 2;
  const originY = (cssH - size) / 2;
  const cell = size / env.gridSize;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cssW, cssH);

  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 2;
  ctx.strokeRect(originX, originY, size, size);

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

  // Comida
  const fx = originX + env.food.x * cell;
  const fy = originY + env.food.y * cell;
  ctx.fillStyle = FOOD;
  ctx.beginPath();
  ctx.arc(fx + cell / 2, fy + cell / 2, cell * 0.28, 0, Math.PI * 2);
  ctx.fill();

  // Cobra
  env.snake.forEach((p, i) => {
    const x = originX + p.x * cell;
    const y = originY + p.y * cell;
    const inset = 2.5;
    roundRect(ctx, x + inset, y + inset, cell - inset * 2, cell - inset * 2, 5);
    ctx.fillStyle = i === 0 ? HEAD : SNAKE;
    ctx.fill();
  });
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
