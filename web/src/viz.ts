/** Visualização ao vivo da rede 11 → 64 → 3 (painel direito). */

import { ACTION_LABELS, FEATURE_LABELS } from "./features";
import type { ForwardResult, MLPQNetwork } from "./network";

const TEAL = { r: 45, g: 212, b: 191 };
const DIM = "#3a3a45";
const TEXT = "#e8e8ef";
const MUTED = "#8a8a9a";

export type VizOptions = {
  hiddenDisplay?: number;
};

function rgba(a: number): string {
  return `rgba(${TEAL.r},${TEAL.g},${TEAL.b},${Math.max(0, Math.min(1, a))})`;
}

export function drawNetwork(
  canvas: HTMLCanvasElement,
  net: MLPQNetwork,
  state: ArrayLike<number>,
  fwd: ForwardResult,
  opts: VizOptions = {},
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
  ctx.clearRect(0, 0, cssW, cssH);

  const hiddenDisplay = opts.hiddenDisplay ?? 14;
  // Índices dos hidden mais ativos (mostra subset).
  const hIdx = Array.from(fwd.h.keys())
    .sort((a, b) => fwd.h[b] - fwd.h[a])
    .slice(0, hiddenDisplay)
    .sort((a, b) => a - b);

  const padX = 18;
  const padY = 28;
  const colX = [padX + 70, cssW * 0.52, cssW - padX - 50];
  const inYs = evenly(FEATURE_LABELS.length, padY, cssH - 20);
  const hYs = evenly(hIdx.length, padY + 10, cssH - 30);
  const outYs = evenly(3, padY + 40, cssH - 50);

  const maxH = Math.max(...Array.from(fwd.h), 1e-6);
  const maxAbsW1 = maxAbs(net.W1);
  const maxAbsW2 = maxAbs(net.W2);

  // Edges input → hidden (só inputs ativos e hidden exibidos).
  for (let i = 0; i < 11; i++) {
    if (state[i] < 0.5) continue;
    for (let hi = 0; hi < hIdx.length; hi++) {
      const j = hIdx[hi];
      const w = net.W1[i][j];
      const contrib = Math.abs(w * state[i]);
      const alpha = 0.08 + 0.55 * (contrib / (maxAbsW1 || 1));
      const width = 0.4 + 2.2 * (contrib / (maxAbsW1 || 1));
      strokeEdge(ctx, colX[0] + 10, inYs[i], colX[1] - 10, hYs[hi], rgba(alpha), width);
    }
  }

  // Edges hidden → output
  for (let hi = 0; hi < hIdx.length; hi++) {
    const j = hIdx[hi];
    if (fwd.h[j] <= 0) continue;
    for (let k = 0; k < 3; k++) {
      const w = net.W2[j][k];
      const contrib = Math.abs(w * fwd.h[j]);
      const alpha = 0.06 + 0.7 * (contrib / ((maxAbsW2 * maxH) || 1));
      const width = 0.5 + 2.8 * (contrib / ((maxAbsW2 * maxH) || 1));
      const chosenBoost = k === fwd.action ? 1.25 : 1;
      strokeEdge(
        ctx,
        colX[1] + 10,
        hYs[hi],
        colX[2] - 14,
        outYs[k],
        rgba(Math.min(1, alpha * chosenBoost)),
        width * chosenBoost,
      );
    }
  }

  // Input nodes + labels
  for (let i = 0; i < 11; i++) {
    const on = state[i] >= 0.5;
    drawNode(ctx, colX[0], inYs[i], on ? 7 : 5, on ? rgba(0.95) : DIM, on);
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = on ? TEXT : MUTED;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(FEATURE_LABELS[i], colX[0] - 14, inYs[i]);
  }

  // Hidden nodes
  for (let hi = 0; hi < hIdx.length; hi++) {
    const j = hIdx[hi];
    const act = fwd.h[j] / maxH;
    const r = 4 + 6 * act;
    drawNode(ctx, colX[1], hYs[hi], r, act > 0 ? rgba(0.35 + 0.65 * act) : DIM, act > 0.05);
  }

  // Output nodes
  const qMax = Math.max(...Array.from(fwd.q), 1e-6);
  const qMin = Math.min(...Array.from(fwd.q));
  for (let k = 0; k < 3; k++) {
    const chosen = k === fwd.action;
    const norm = (fwd.q[k] - qMin) / (qMax - qMin + 1e-9);
    const r = chosen ? 14 : 7 + 4 * norm;
    drawNode(ctx, colX[2], outYs[k], r, chosen ? rgba(1) : DIM, chosen);
    ctx.font = chosen ? "bold 13px ui-sans-serif, system-ui, sans-serif" : "12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = chosen ? TEXT : MUTED;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${ACTION_LABELS[k]}  ${fwd.q[k].toFixed(2)}`, colX[2] + r + 10, outYs[k]);
  }
}

function evenly(n: number, top: number, bottom: number): number[] {
  if (n <= 1) return [(top + bottom) / 2];
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(top + ((bottom - top) * i) / (n - 1));
  return out;
}

function maxAbs(W: number[][]): number {
  let m = 0;
  for (const row of W) for (const v of row) m = Math.max(m, Math.abs(v));
  return m || 1;
}

function strokeEdge(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const cx = (x1 + x2) / 2;
  ctx.bezierCurveTo(cx, y1, cx, y2, x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawNode(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  glow: boolean,
): void {
  if (glow) {
    ctx.beginPath();
    ctx.arc(x, y, r + 4, 0, Math.PI * 2);
    ctx.fillStyle = rgba(0.15);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = glow ? rgba(0.9) : "#555";
  ctx.lineWidth = 1.2;
  ctx.stroke();
}
