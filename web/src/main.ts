import "./styles.css";
import { ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, DIR, SnakeEnv, type Direction } from "./env";
import { extractState } from "./features";
import { loadModel, MLPQNetwork, ReplayBuffer } from "./network";
import { drawGame } from "./renderGame";
import { drawNetwork } from "./viz";

const gameCanvas = document.querySelector<HTMLCanvasElement>("#gameCanvas")!;
const netCanvas = document.querySelector<HTMLCanvasElement>("#netCanvas")!;
const scoreEl = document.querySelector<HTMLElement>("#score")!;
const bestEl = document.querySelector<HTMLElement>("#best")!;
const avgScoreEl = document.querySelector<HTMLElement>("#avgScore")!;
const gamesCountEl = document.querySelector<HTMLElement>("#gamesCount")!;
const seedLabel = document.querySelector<HTMLElement>("#seedLabel")!;
const statusLabel = document.querySelector<HTMLElement>("#statusLabel")!;
const speedInput = document.querySelector<HTMLInputElement>("#speed")!;
const speedLabel = document.querySelector<HTMLElement>("#speedLabel")!;
const btnNew = document.querySelector<HTMLButtonElement>("#btnNew")!;
const btnPlay = document.querySelector<HTMLButtonElement>("#btnPlay")!;
const btnStep = document.querySelector<HTMLButtonElement>("#btnStep")!;
const btnModeTrained = document.querySelector<HTMLButtonElement>("#btnModeTrained")!;
const btnModeHuman = document.querySelector<HTMLButtonElement>("#btnModeHuman")!;
const btnModeRandom = document.querySelector<HTMLButtonElement>("#btnModeRandom")!;
const btnAutoRestart = document.querySelector<HTMLButtonElement>("#btnAutoRestart")!;
const btnTrain = document.querySelector<HTMLButtonElement>("#btnTrain")!;
const curveSection = document.querySelector<HTMLElement>("#curveSection")!;
const curveCanvas = document.querySelector<HTMLCanvasElement>("#curveCanvas")!;
const curveCaption = document.querySelector<HTMLElement>("#curveCaption")!;

type PlayMode = "trained" | "human" | "random" | "browser_trained";

let trainedNet: MLPQNetwork | null = null;
let browserNet: MLPQNetwork | null = null;
let liveNet: MLPQNetwork | null = null;
let mode: PlayMode = "trained";
let autoRestart = true;
let env = new SnakeEnv(10, 13);
let seed = 13;
let playing = true;
let bestScore = 0;
let sessionScores: number[] = [];
let lastTs = 0;
let acc = 0;
let training = false;
let pendingHumanDir: Direction | null = null;
let restartTimer: number | null = null;
let cachedCurve: { episode: number; score: number; ma_score: number | null }[] = [];

function activeNet(): MLPQNetwork {
  if (!liveNet) throw new Error("rede não carregada");
  return liveNet;
}

function updateModeButtons(): void {
  btnModeTrained.classList.toggle("active", mode === "trained" || mode === "browser_trained");
  btnModeTrained.textContent = mode === "browser_trained" ? "IA (Treino JS)" : "IA Treinada";
  btnModeHuman.classList.toggle("active", mode === "human");
  btnModeRandom.classList.toggle("active", mode === "random");
}

function syncModeNet(): void {
  if (!trainedNet) return;
  if (mode === "trained") {
    liveNet = trainedNet.clone();
  } else if (mode === "browser_trained" && browserNet) {
    liveNet = browserNet.clone();
  } else if (mode === "random") {
    liveNet = MLPQNetwork.random((Date.now() % 1e9) | 0);
  } else {
    // Modo humano mostra a rede treinada avaliando cada estado em tempo real
    liveNet = (browserNet ?? trainedNet).clone();
  }
  updateModeButtons();
}

function clearRestartTimer(): void {
  if (restartTimer !== null) {
    window.clearTimeout(restartTimer);
    restartTimer = null;
  }
}

function recordFinishedGame(finalScore: number): void {
  sessionScores.push(finalScore);
  if (finalScore > bestScore) {
    bestScore = finalScore;
    bestEl.textContent = String(bestScore);
  }
  const avg = sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length;
  avgScoreEl.textContent = avg.toFixed(1);
  gamesCountEl.textContent = String(sessionScores.length);
}

function newGame(nextSeed?: number): void {
  clearRestartTimer();
  seed = nextSeed ?? ((Math.random() * 1e9) | 0);
  env = new SnakeEnv(10, seed);
  pendingHumanDir = null;
  seedLabel.textContent = String(seed);
  scoreEl.textContent = "0";
  statusLabel.textContent = mode === "human" ? "jogando (use WASD/Setas)" : "jogando";
  render();
}

function relativeActionFromTargetDir(current: Direction, target: Direction): number {
  if (target === current) return ACTION_STRAIGHT;
  if (target === (((current + 1) % 4) as Direction)) return ACTION_RIGHT;
  if (target === (((current + 3) % 4) as Direction)) return ACTION_LEFT;
  // Sentido oposto (180 graus) ignorado para não se auto-colidir para trás
  return ACTION_STRAIGHT;
}

function tick(): void {
  if (env.done) return;

  const state = extractState(env);
  const fwd = activeNet().forward(state);

  let action = fwd.action;
  if (mode === "human") {
    if (pendingHumanDir !== null) {
      action = relativeActionFromTargetDir(env.direction, pendingHumanDir);
      pendingHumanDir = null;
    } else {
      action = ACTION_STRAIGHT;
    }
  }

  env.step(action);
  scoreEl.textContent = String(env.score);
  if (env.score > bestScore) {
    bestScore = env.score;
    bestEl.textContent = String(bestScore);
  }

  if (env.done) {
    recordFinishedGame(env.score);
    statusLabel.textContent = `fim · score ${env.score}`;
    render();
    if (autoRestart && playing && !training) {
      restartTimer = window.setTimeout(() => {
        if (env.done && playing && !training) newGame();
      }, 950);
    }
    return;
  }

  render();
}

function render(): void {
  const state = extractState(env);
  const fwd = activeNet().forward(state);
  drawGame(gameCanvas, env, {
    humanMode: mode === "human",
    aiSuggestedAction: fwd.action,
  });
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
  if (playing) {
    if (env.done) newGame();
    else statusLabel.textContent = "jogando";
  } else {
    clearRestartTimer();
    statusLabel.textContent = "pausado";
  }
});
btnStep.addEventListener("click", () => {
  clearRestartTimer();
  playing = false;
  btnPlay.textContent = "Play";
  btnPlay.classList.remove("active");
  if (env.done) newGame();
  else tick();
});
speedInput.addEventListener("input", () => {
  speedLabel.textContent = `${speedInput.value}×`;
});

function setMode(nextMode: PlayMode): void {
  mode = nextMode;
  if (mode === "human" && Number(speedInput.value) > 8) {
    speedInput.value = "6";
    speedLabel.textContent = "6×";
  }
  syncModeNet();
  newGame(seed);
}

btnModeTrained.addEventListener("click", () => setMode(browserNet ? "browser_trained" : "trained"));
btnModeHuman.addEventListener("click", () => setMode("human"));
btnModeRandom.addEventListener("click", () => setMode("random"));

btnAutoRestart.addEventListener("click", () => {
  autoRestart = !autoRestart;
  btnAutoRestart.textContent = `Auto-reiniciar: ${autoRestart ? "ON" : "OFF"}`;
  btnAutoRestart.classList.toggle("active", autoRestart);
});

window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement?.tagName !== "BUTTON") {
    e.preventDefault();
    if (env.done) newGame();
    else btnPlay.click();
    return;
  }

  const keyMap: Record<string, Direction> = {
    ArrowUp: DIR.UP,
    KeyW: DIR.UP,
    ArrowDown: DIR.DOWN,
    KeyS: DIR.DOWN,
    ArrowLeft: DIR.LEFT,
    KeyA: DIR.LEFT,
    ArrowRight: DIR.RIGHT,
    KeyD: DIR.RIGHT,
  };
  const wantedDir = keyMap[e.code];
  if (wantedDir !== undefined) {
    if (mode !== "human") {
      mode = "human";
      if (Number(speedInput.value) > 8) {
        speedInput.value = "6";
        speedLabel.textContent = "6×";
      }
      syncModeNet();
      statusLabel.textContent = "modo teclado ativado (WASD/Setas)";
    }
    e.preventDefault();
    pendingHumanDir = wantedDir;
  }
});

btnTrain.addEventListener("click", async () => {
  if (training) return;
  clearRestartTimer();
  training = true;
  playing = false;
  btnPlay.textContent = "Play";
  btnPlay.classList.remove("active");
  btnTrain.disabled = true;

  const online = MLPQNetwork.random(42);
  const targetNet = online.clone();
  const replay = new ReplayBuffer(20_000);
  liveNet = online;
  mode = "browser_trained";
  updateModeButtons();

  const episodes = 260;
  const batchSize = 32;
  const minReplay = 96;
  const targetUpdateEvery = 60;
  const gamma = 0.9;
  const lr = 0.0025;
  let eps = 1.0;
  let trainSteps = 0;
  const scores: number[] = [];
  const curveRecords: { episode: number; score: number; ma_score: number | null }[] = [];

  for (let ep = 1; ep <= episodes; ep++) {
    const epEnv = new SnakeEnv(10, ep * 9973 + 42);
    let state = extractState(epEnv);
    while (!epEnv.done) {
      const fwd = online.forward(state);
      const action = Math.random() < eps ? ((Math.random() * 3) | 0) : fwd.action;
      const prevDist =
        Math.abs(epEnv.snake[0].x - epEnv.food.x) + Math.abs(epEnv.snake[0].y - epEnv.food.y);
      const result = epEnv.step(action);
      const newDist =
        Math.abs(epEnv.snake[0].x - epEnv.food.x) + Math.abs(epEnv.snake[0].y - epEnv.food.y);
      const shaping = !result.done && result.reward === 0 ? (newDist < prevDist ? 0.15 : -0.2) : 0;
      const next = extractState(epEnv);

      replay.push({
        state,
        action,
        reward: result.reward + shaping,
        nextState: next,
        done: result.done,
      });

      if (replay.length >= minReplay) {
        const batch = replay.sample(batchSize);
        const bStates: Float64Array[] = [];
        const bActions: number[] = [];
        const bTargets: number[] = [];
        for (const item of batch) {
          const nextQ = targetNet.forward(item.nextState).q;
          const maxNext = Math.max(nextQ[0], nextQ[1], nextQ[2]);
          const y = item.done ? item.reward : item.reward + gamma * maxNext;
          bStates.push(item.state);
          bActions.push(item.action);
          bTargets.push(y);
        }
        online.trainBatch(bStates, bActions, bTargets, lr);
        trainSteps++;
        if (trainSteps % targetUpdateEvery === 0) {
          targetNet.copyWeightsFrom(online);
        }
      }
      state = next;
    }

    scores.push(epEnv.score);
    const winSlice = scores.slice(Math.max(0, scores.length - 20));
    const ma = winSlice.reduce((a, b) => a + b, 0) / winSlice.length;
    curveRecords.push({ episode: ep, score: epEnv.score, ma_score: ma });
    eps = Math.max(0.02, eps * 0.982);

    if (ep % 10 === 0 || ep === episodes) {
      statusLabel.textContent = `treinando DQN no browser… ep ${ep}/${episodes} · média20=${ma.toFixed(1)}`;
      env = epEnv;
      scoreEl.textContent = String(epEnv.score);
      cachedCurve = curveRecords;
      curveSection.hidden = false;
      curveCaption.textContent = `Treino ao vivo no navegador (Replay Buffer + Target Net) · média móvel janela=20`;
      drawCurve(curveCanvas, cachedCurve);
      render();
      await wait(0);
    }
  }

  browserNet = online.clone();
  liveNet = browserNet.clone();
  const maxTrained = Math.max(...scores);
  if (maxTrained > bestScore) {
    bestScore = maxTrained;
    bestEl.textContent = String(bestScore);
  }
  statusLabel.textContent = `treino JS concluído · melhor ${maxTrained}`;
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
    cachedCurve = data.episodes;
    curveSection.hidden = false;
    curveCaption.textContent = `run ${data.run_id} · média móvel janela=${data.window} · ${data.episodes.length} episódios`;
    drawCurve(curveCanvas, cachedCurve);
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

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 1000;
  const cssH = canvas.clientHeight || 180;
  if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#0a0a0e";
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 38;
  const padR = 16;
  const padT = 18;
  const padB = 24;
  const plotW = Math.max(10, cssW - padL - padR);
  const plotH = Math.max(10, cssH - padT - padB);

  const maxScore = Math.max(...episodes.map((e) => e.score), 5);
  const maxEp = Math.max(episodes[episodes.length - 1].episode, 1);

  const xOf = (ep: number) => padL + (ep / maxEp) * plotW;
  const yOf = (s: number) => padT + plotH - (s / maxScore) * plotH;

  // Linhas horizontais de grade + labels de eixo Y
  ctx.font = "10px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = [0, Math.round(maxScore / 2), maxScore];
  for (const val of yTicks) {
    const y = yOf(val);
    ctx.strokeStyle = "#191923";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(cssW - padR, y);
    ctx.stroke();

    ctx.fillStyle = "#8a8a9a";
    ctx.fillText(String(val), padL - 6, y);
  }

  // Labels do eixo X
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = [1, Math.round(maxEp / 2), maxEp];
  for (const ep of xTicks) {
    ctx.fillStyle = "#8a8a9a";
    ctx.fillText(`ep ${ep}`, xOf(ep), cssH - padB + 6);
  }

  // Pontos de score bruto
  ctx.fillStyle = "rgba(45, 212, 191, 0.35)";
  for (const e of episodes) {
    ctx.beginPath();
    ctx.arc(xOf(e.episode), yOf(e.score), 1.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Linha da média móvel
  ctx.beginPath();
  ctx.strokeStyle = "#f5b041";
  ctx.lineWidth = 2.2;
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

  // Legenda no canto superior esquerdo
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(45, 212, 191, 0.85)";
  ctx.fillText("● score do episódio", padL + 6, 6);
  ctx.fillStyle = "#f5b041";
  ctx.fillText("━ média móvel", padL + 135, 6);
}

window.addEventListener("resize", () => {
  render();
  if (cachedCurve.length > 0) drawCurve(curveCanvas, cachedCurve);
});

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
