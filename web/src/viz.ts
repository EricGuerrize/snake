/** Visualização ao vivo da rede 11 → 64 → 3 (painel direito). */

import { ACTION_LABELS, FEATURE_LABELS } from "./features";
import type { ForwardResult, MLPQNetwork } from "./network";

const TEAL = { r: 45, g: 212, b: 191 };
const CORAL = { r: 248, g: 113, b: 113 };
const DIM = "#3a3a45";
const TEXT = "#e8e8ef";
const MUTED = "#8a8a9a";

export type VizOptions = {
  hiddenDisplay?: number;
};

function rgbaTeal(a: number): string {
  return `rgba(${TEAL.r},${TEAL.g},${TEAL.b},${Math.max(0, Math.min(1, a))})`;
}

function rgbaSigned(val: number, a: number): string {
  const c = val >= 0 ? TEAL : CORAL;
  return `rgba(${c.r},${c.g},${c.b},${Math.max(0, Math.min(1, a))})`;
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
  const topHeader = 36;
  const bottomLegend = 28;
  const colX = [padX + 86, cssW * 0.5, cssW - padX - 88];
  const inYs = evenly(FEATURE_LABELS.length, topHeader, cssH - bottomLegend);
  const hYs = evenly(hIdx.length, topHeader + 6, cssH - bottomLegend - 6);
  const outYs = evenly(3, topHeader + 48, cssH - bottomLegend - 48);

  // Títulos das camadas
  ctx.font = "600 10.5px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.fillStyle = MUTED;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText("ENTRADA (11)", colX[0] - 24, 15);
  ctx.fillText(`OCULTA ReLU (top ${hIdx.length}/64)`, colX[1], 15);
  ctx.fillText("SAÍDA Q(s,a)", colX[2] + 24, 15);

  const maxH = Math.max(...Array.from(fwd.h), 1e-6);
  const maxAbsW1 = maxAbs(net.W1);
  const maxAbsW2 = maxAbs(net.W2);

  // Edges input → hidden (só inputs ativos e hidden exibidos).
  for (let i = 0; i < 11; i++) {
    if (state[i] < 0.5) continue;
    for (let hi = 0; hi < hIdx.length; hi++) {
      const j = hIdx[hi];
      const w = net.W1[i][j];
      const signal = w * state[i];
      const contrib = Math.abs(signal);
      const alpha = 0.08 + 0.58 * (contrib / (maxAbsW1 || 1));
      const width = 0.45 + 2.1 * (contrib / (maxAbsW1 || 1));
      strokeEdge(
        ctx,
        colX[0] + 10,
        inYs[i],
        colX[1] - 10,
        hYs[hi],
        rgbaSigned(signal, alpha),
        width,
      );
    }
  }

  // Edges hidden → output
  for (let hi = 0; hi < hIdx.length; hi++) {
    const j = hIdx[hi];
    if (fwd.h[j] <= 0) continue;
    for (let k = 0; k < 3; k++) {
      const w = net.W2[j][k];
      const signal = w * fwd.h[j];
      const contrib = Math.abs(signal);
      const alpha = 0.08 + 0.72 * (contrib / (maxAbsW2 * maxH || 1));
      const width = 0.5 + 2.7 * (contrib / (maxAbsW2 * maxH || 1));
      const chosenBoost = k === fwd.action ? 1.25 : 0.95;
      strokeEdge(
        ctx,
        colX[1] + 10,
        hYs[hi],
        colX[2] - 14,
        outYs[k],
        rgbaSigned(signal, Math.min(1, alpha * chosenBoost)),
        width * chosenBoost,
      );
    }
  }

  // Input nodes + labels
  for (let i = 0; i < 11; i++) {
    const on = state[i] >= 0.5;
    const isDangerOn = on && i < 3;
    const fill = isDangerOn ? "rgba(248,113,113,0.95)" : on ? rgbaTeal(0.95) : DIM;
    drawNode(ctx, colX[0], inYs[i], on ? 6.5 : 4.8, fill, on, isDangerOn);
    ctx.font = on
      ? "600 11.5px 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif"
      : "11.5px 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = isDangerOn ? "#fca5a5" : on ? TEXT : MUTED;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(FEATURE_LABELS[i], colX[0] - 13, inYs[i]);
  }

  // Hidden nodes + neuron id
  for (let hi = 0; hi < hIdx.length; hi++) {
    const j = hIdx[hi];
    const act = fwd.h[j] / maxH;
    const r = 3.8 + 5.5 * act;
    drawNode(ctx, colX[1], hYs[hi], r, act > 0 ? rgbaTeal(0.35 + 0.65 * act) : DIM, act > 0.08);
    if (act > 0.25) {
      ctx.font = "9.5px 'IBM Plex Mono', ui-monospace, monospace";
      ctx.fillStyle = "rgba(138, 138, 154, 0.7)";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`h${j}`, colX[1] + r + 5, hYs[hi]);
    }
  }

  // Output nodes
  const qMax = Math.max(...Array.from(fwd.q), 1e-6);
  const qMin = Math.min(...Array.from(fwd.q));
  for (let k = 0; k < 3; k++) {
    const chosen = k === fwd.action;
    const norm = (fwd.q[k] - qMin) / (qMax - qMin + 1e-9);
    const r = chosen ? 13 : 6.5 + 3.5 * norm;
    drawNode(ctx, colX[2], outYs[k], r, chosen ? rgbaTeal(1) : DIM, chosen);
    ctx.font = chosen
      ? "700 12.5px 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif"
      : "12px 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = chosen ? TEXT : MUTED;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${ACTION_LABELS[k]}`, colX[2] + r + 10, outYs[k] - 7);
    ctx.font = "11px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = chosen ? "#2dd4bf" : MUTED;
    ctx.fillText(`Q = ${fwd.q[k].toFixed(2)}`, colX[2] + r + 10, outYs[k] + 8);
  }

  // Legenda de sinais no rodapé do painel
  const legY = cssH - 12;
  ctx.font = "10.5px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(45, 212, 191, 0.9)";
  ctx.fillText("● peso + (excita)", padX, legY);
  ctx.fillStyle = "rgba(248, 113, 113, 0.9)";
  ctx.fillText("● peso − (inibe)", padX + 122, legY);
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
  dangerGlow = false,
): void {
  if (glow) {
    ctx.beginPath();
    ctx.arc(x, y, r + 4, 0, Math.PI * 2);
    ctx.fillStyle = dangerGlow ? "rgba(248, 113, 113, 0.2)" : rgbaTeal(0.16);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = glow ? (dangerGlow ? "rgba(248, 113, 113, 0.9)" : rgbaTeal(0.9)) : "#555";
  ctx.lineWidth = 1.2;
  ctx.stroke();
}
