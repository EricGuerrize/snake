/** Ambiente Snake headless — espelha snake_rl/env.py (regras + seed local). */

export const DIR = { RIGHT: 0, DOWN: 1, LEFT: 2, UP: 3 } as const;
export type Direction = (typeof DIR)[keyof typeof DIR];

export const ACTION_STRAIGHT = 0;
export const ACTION_RIGHT = 1;
export const ACTION_LEFT = 2;

export const REWARD_FOOD = 10;
export const REWARD_CRASH = -10;

export type Point = { x: number; y: number };

export type StepResult = {
  reward: number;
  done: boolean;
  score: number;
  info: Record<string, string>;
};

/** PRNG mulberry32 — determinístico por seed (não precisa bater com NumPy). */
export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export class SnakeEnv {
  gridSize: number;
  snake: Point[] = [];
  direction: Direction = DIR.RIGHT;
  food: Point = { x: 0, y: 0 };
  score = 0;
  steps = 0;
  done = false;
  maxStepsWithoutFood: number;
  private stepsSinceFood = 0;
  private rng: () => number;

  constructor(gridSize = 10, seed = 0) {
    if (gridSize < 4) throw new Error("gridSize deve ser >= 4");
    this.gridSize = gridSize;
    this.maxStepsWithoutFood = gridSize * gridSize * 2;
    this.rng = mulberry32(seed);
    this.reset(seed);
  }

  reset(seed?: number): void {
    if (seed !== undefined) this.rng = mulberry32(seed);
    const mid = Math.floor(this.gridSize / 2);
    this.snake = [
      { x: mid, y: mid },
      { x: mid - 1, y: mid },
      { x: mid - 2, y: mid },
    ];
    this.direction = DIR.RIGHT;
    this.score = 0;
    this.steps = 0;
    this.done = false;
    this.stepsSinceFood = 0;
    this.placeFood();
  }

  static turn(direction: Direction, action: number): Direction {
    if (action === ACTION_STRAIGHT) return direction;
    if (action === ACTION_RIGHT) return ((direction + 1) % 4) as Direction;
    if (action === ACTION_LEFT) return ((direction + 3) % 4) as Direction;
    throw new Error(`ação inválida: ${action}`);
  }

  static nextCell(pos: Point, direction: Direction): Point {
    if (direction === DIR.RIGHT) return { x: pos.x + 1, y: pos.y };
    if (direction === DIR.LEFT) return { x: pos.x - 1, y: pos.y };
    if (direction === DIR.UP) return { x: pos.x, y: pos.y - 1 };
    return { x: pos.x, y: pos.y + 1 };
  }

  isCollision(pos: Point, body: Point[] = this.snake): boolean {
    if (pos.x < 0 || pos.y < 0 || pos.x >= this.gridSize || pos.y >= this.gridSize) {
      return true;
    }
    return body.some((p) => p.x === pos.x && p.y === pos.y);
  }

  private placeFood(): void {
    const occupied = new Set(this.snake.map((p) => `${p.x},${p.y}`));
    const free: Point[] = [];
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        if (!occupied.has(`${x},${y}`)) free.push({ x, y });
      }
    }
    if (free.length === 0) {
      this.food = { ...this.snake[0] };
      return;
    }
    const idx = Math.floor(this.rng() * free.length);
    this.food = free[idx];
  }

  step(action: number): StepResult {
    if (this.done) throw new Error("episódio encerrado — chame reset()");

    this.direction = SnakeEnv.turn(this.direction, action);
    const newHead = SnakeEnv.nextCell(this.snake[0], this.direction);
    this.steps += 1;
    this.stepsSinceFood += 1;

    const willEat = newHead.x === this.food.x && newHead.y === this.food.y;
    const bodyToCheck = willEat ? this.snake : this.snake.slice(0, -1);
    if (this.isCollision(newHead, bodyToCheck)) {
      this.done = true;
      return { reward: REWARD_CRASH, done: true, score: this.score, info: { reason: "crash" } };
    }

    this.snake.unshift(newHead);
    let reward = 0;
    if (willEat) {
      this.score += 1;
      this.stepsSinceFood = 0;
      this.placeFood();
      reward = REWARD_FOOD;
      if (this.snake.length >= this.gridSize * this.gridSize) {
        this.done = true;
        return { reward, done: true, score: this.score, info: { reason: "win" } };
      }
    } else {
      this.snake.pop();
    }

    if (this.stepsSinceFood >= this.maxStepsWithoutFood) {
      this.done = true;
      return { reward: REWARD_CRASH, done: true, score: this.score, info: { reason: "timeout" } };
    }

    return { reward, done: false, score: this.score, info: {} };
  }
}
