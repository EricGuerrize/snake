import "./styles.css";
import { SnakeEnv } from "./env";
import { extractState } from "./features";
import { loadModel, MLPQNetwork } from "./network";
import { drawGame } from "./renderGame";
import { drawNetwork } from "./viz";

const gameCanvas = document.querySelector<HTMLCanvasElement>("#gameCanvas")!;
const netCanvas = document.querySelector<HTMLCanvasElement>("#netCanvas")!;
const scoreEl = document.querySelector<HTMLElement>("#score")!;
const bestEl = document.querySelector<HTMLElement>("#best")!;
const seedLabel = document.querySelector<HTMLElement>("#seedLabel")!;
const statusLabel = document.querySelector<HTMLElement>("#statusLabel")!;
const speedInput = document.querySelector<HTMLInputElement>("#speed")!;
const speedLabel = document.querySelector<HTMLElement>("#speedLabel")!;
const btnNew = document.querySelector<HTMLButtonElement>("#btnNew")!;
const btnPlay = document.querySelector<HTMLButtonElement>("#btnPlay")!;
const btnStep = document.querySelector<HTMLButtonElement>("#btnStep")!;
const btnMode = document.querySelector<HTMLButtonElement>("#btnMode")!;
const btnTrain = document.querySelector<HTMLButtonElement>("#btnTrain")!;
const curveSection = document.querySelector<HTMLElement>("#curveSection")!;
const curveCanvas = document.querySelector<HTMLCanvasElement>("#curveCanvas")!;
const curveCaption = document.querySelector<HTMLElement>("#curveCaption")!;

let trainedNet: MLPQNetwork | null = null;
let liveNet: MLPQNetwork | null = null;
let useTrained = true;
let env = new SnakeEnv(10, 13);
let seed = 13;
let playing = true;
let bestScore = 0;
let lastTs = 0;
let acc = 0;
let training = false;

function activeNet(): MLPQNetwork {
  if (!liveNet) throw new Error("rede não carregada");
  return liveNet;
}

function syncModeNet(): void {
  if (!trainedNet) return;
  liveNet = useTrained ? trainedNet.clone() : MLPQNetwork.random(Date.now() % 1e9);
  btnMode.textContent = useTrained ? "Agente treinado" : "Agente aleatório";
  btnMode.classList.toggle("active", useTrained);
}

function newGame(nextSeed?: number): void {
  seed = nextSeed ?? ((Math.random() * 1e9) | 0);
  env = new SnakeEnv(10, seed);
  seedLabel.textContent = String(seed);
  scoreEl.textContent = "0";
  statusLabel.textContent = "jogando";
  render();
}

function tick(): void {
  if (env.done) {
    statusLabel.textContent = `fim · score ${env.score}`;
    return;
  }
  const state = extractState(env);
  const fwd = activeNet().forward(state);
  env.step(fwd.action);
  scoreEl.textContent = String(env.score);
  if (env.score > bestScore) {
    bestScore = env.score;
    bestEl.textContent = String(bestScore);
  }
  if (env.done) statusLabel.textContent = `fim · score ${env.score}`;
  render();
}

function render(): void {
  const state = extractState(env);
  const fwd = activeNet().forward(state);
  drawGame(gameCanvas, env);
  drawNetwork(netCanvas, activeNet(), state, fwd);
}

function loop(ts: number): void {
  const dt = ts - lastTs;
  lastTs = ts;
  if (playing && !training && !env.done) {
    const interval = 1000 / Math.max(1, Number(speedInput.value));
    acc += dt;
    while (acc >= interval) {
      acc -= interval;
      tick();
      if (env.done) break;
    }
  }
  requestAnimationFrame(loop);
}

btnNew.addEventListener("click", () => newGame());
btnPlay.addEventListener("click", () => {
  playing = !playing;
  btnPlay.textContent = playing ? "Pausar" : "Play";
  btnPlay.classList.toggle("active", playing);
  if (playing) statusLabel.textContent = env.done ? `fim · score ${env.score}` : "jogando";
  else statusLabel.textContent = "pausado";
});
btnStep.addEventListener("click", () => {
  playing = false;
  btnPlay.textContent = "Play";
  btnPlay.classList.remove("active");
  tick();
});
speedInput.addEventListener("input", () => {
  speedLabel.textContent = `${speedInput.value}×`;
});
btnMode.addEventListener("click", () => {
  useTrained = !useTrained;
  syncModeNet();
  newGame(seed);
});

btnTrain.addEventListener("click", async () => {
  if (training) return;
  training = true;
  playing = false;
  btnPlay.textContent = "Play";
  btnPlay.classList.remove("active");
  btnTrain.disabled = true;
  useTrained = false;
  liveNet = MLPQNetwork.random(7);
  btnMode.textContent = "Agente (treino JS)";
  btnMode.classList.remove("active");

  const episodes = 120;
  const gamma = 0.9;
  const lr = 0.002;
  let eps = 1.0;
  const scores: number[] = [];

  for (let ep = 1; ep <= episodes; ep++) {
    statusLabel.textContent = `treinando JS… ep ${ep}/${episodes}`;
    const epEnv = new SnakeEnv(10, ep * 9973 + 42);
    let state = extractState(epEnv);
    while (!epEnv.done) {
      const fwd = liveNet.forward(state);
      const action = Math.random() < eps ? ((Math.random() * 3) | 0) : fwd.action;
      const result = epEnv.step(action);
      const next = extractState(epEnv);
      const nextQ = liveNet.forward(next).q;
      const maxNext = Math.max(nextQ[0], nextQ[1], nextQ[2]);
      const target = result.done ? result.reward : result.reward + gamma * maxNext;
      liveNet.trainStep(state, action, target, lr);
      state = next;
    }
    scores.push(epEnv.score);
    eps = Math.max(0.05, eps * 0.97);
    if (ep % 5 === 0) {
      env = epEnv;
      scoreEl.textContent = String(epEnv.score);
      render();
      await wait(0); // yield to UI
    }
  }

  bestScore = Math.max(bestScore, ...scores);
  bestEl.textContent = String(bestScore);
  statusLabel.textContent = `treino JS ok · melhor ${Math.max(...scores)}`;
  training = false;
  btnTrain.disabled = false;
  newGame();
  playing = true;
  btnPlay.textContent = "Pausar";
  btnPlay.classList.add("active");
});

function wait(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function loadCurve(): Promise<void> {
  try {
    const res = await fetch("./learning_curve.json");
    if (!res.ok) return;
    const data = (await res.json()) as {
      run_id: string;
      window: number;
      episodes: { episode: number; score: number; ma_score: number | null }[];
    };
    curveSection.hidden = false;
    curveCaption.textContent = `run ${data.run_id} · média móvel janela=${data.window}`;
    drawCurve(curveCanvas, data.episodes);
  } catch {
    // opcional
  }
}

function drawCurve(
  canvas: HTMLCanvasElement,
  episodes: { episode: number; score: number; ma_score: number | null }[],
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || episodes.length === 0) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0a0e";
  ctx.fillRect(0, 0, w, h);

  const maxScore = Math.max(...episodes.map((e) => e.score), 1);
  const maxEp = episodes[episodes.length - 1].episode;

  const xOf = (ep: number) => (ep / maxEp) * (w - 20) + 10;
  const yOf = (s: number) => h - 12 - (s / maxScore) * (h - 28);

  ctx.fillStyle = "rgba(45,212,191,0.35)";
  for (const e of episodes) {
    ctx.fillRect(xOf(e.episode) - 0.6, yOf(e.score), 1.2, 1.2);
  }

  ctx.beginPath();
  ctx.strokeStyle = "#f5b041";
  ctx.lineWidth = 2;
  let started = false;
  for (const e of episodes) {
    if (e.ma_score == null) continue;
    const x = xOf(e.episode);
    const y = yOf(e.ma_score);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

window.addEventListener("resize", () => render());

async function boot(): Promise<void> {
  statusLabel.textContent = "carregando modelo…";
  trainedNet = await loadModel("./model.json");
  syncModeNet();
  newGame(13);
  await loadCurve();
  lastTs = performance.now();
  requestAnimationFrame(loop);
  statusLabel.textContent = "jogando";
}

boot().catch((err) => {
  console.error(err);
  statusLabel.textContent = String(err);
});
