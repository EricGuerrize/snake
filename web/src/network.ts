/** MLP 11 → 64 ReLU → 3 — forward (+ backprop opcional) em JS puro. */

export type ModelWeights = {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  input_size?: number;
  hidden_size?: number;
  output_size?: number;
};

export type ForwardResult = {
  q: Float64Array;
  h: Float64Array;
  action: number;
};

function matVec(W: number[][], x: ArrayLike<number>, b: number[]): Float64Array {
  const out = new Float64Array(W[0].length);
  for (let j = 0; j < out.length; j++) {
    let sum = b[j];
    for (let i = 0; i < W.length; i++) sum += x[i] * W[i][j];
    out[j] = sum;
  }
  return out;
}

export class MLPQNetwork {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  inputSize: number;
  hiddenSize: number;
  outputSize: number;

  constructor(weights: ModelWeights) {
    this.W1 = weights.W1.map((row) => row.slice());
    this.b1 = weights.b1.slice();
    this.W2 = weights.W2.map((row) => row.slice());
    this.b2 = weights.b2.slice();
    this.inputSize = weights.input_size ?? this.W1.length;
    this.hiddenSize = weights.hidden_size ?? this.b1.length;
    this.outputSize = weights.output_size ?? this.b2.length;
  }

  static random(seed = 1, input = 11, hidden = 64, output = 3): MLPQNetwork {
    let s = seed >>> 0;
    const randn = () => {
      // Box-Muller simplificado via LCG
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      const u1 = (s >>> 0) / 4294967296 || 1e-12;
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      const u2 = (s >>> 0) / 4294967296;
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const scale1 = Math.sqrt(2 / input);
    const scale2 = Math.sqrt(2 / hidden);
    const W1 = Array.from({ length: input }, () =>
      Array.from({ length: hidden }, () => randn() * scale1),
    );
    const W2 = Array.from({ length: hidden }, () =>
      Array.from({ length: output }, () => randn() * scale2),
    );
    return new MLPQNetwork({
      W1,
      b1: Array(hidden).fill(0),
      W2,
      b2: Array(output).fill(0),
      input_size: input,
      hidden_size: hidden,
      output_size: output,
    });
  }

  forward(x: ArrayLike<number>): ForwardResult {
    const z1 = matVec(this.W1, x, this.b1);
    const h = new Float64Array(z1.length);
    for (let i = 0; i < z1.length; i++) h[i] = Math.max(0, z1[i]);
    const q = matVec(this.W2, h, this.b2);
    let action = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[action]) action = i;
    return { q, h, action };
  }

  /** Um passo SGD MSE em um único exemplo (para treino no browser). */
  trainStep(
    x: ArrayLike<number>,
    action: number,
    target: number,
    lr: number,
  ): { loss: number; result: ForwardResult } {
    const result = this.forward(x);
    const pred = result.q[action];
    const residual = pred - target;
    const loss = residual * residual;

    const dq = new Float64Array(this.outputSize);
    dq[action] = 2 * residual;

    // dW2, db2
    for (let j = 0; j < this.outputSize; j++) {
      this.b2[j] -= lr * dq[j];
      for (let i = 0; i < this.hiddenSize; i++) {
        this.W2[i][j] -= lr * result.h[i] * dq[j];
      }
    }

    // dh via ReLU
    const dh = new Float64Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.outputSize; j++) sum += dq[j] * this.W2[i][j];
      dh[i] = result.h[i] > 0 ? sum : 0;
    }

    for (let i = 0; i < this.hiddenSize; i++) {
      this.b1[i] -= lr * dh[i];
      for (let k = 0; k < this.inputSize; k++) {
        this.W1[k][i] -= lr * x[k] * dh[i];
      }
    }

    return { loss, result };
  }

  clone(): MLPQNetwork {
    return new MLPQNetwork({
      W1: this.W1,
      b1: this.b1,
      W2: this.W2,
      b2: this.b2,
      input_size: this.inputSize,
      hidden_size: this.hiddenSize,
      output_size: this.outputSize,
    });
  }
}

export async function loadModel(url = "./model.json"): Promise<MLPQNetwork> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Falha ao carregar ${url}: ${res.status}`);
  const data = (await res.json()) as ModelWeights;
  return new MLPQNetwork(data);
}
