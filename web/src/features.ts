/** Vetor de 11 features — espelha snake_rl/features.py. */

import {
  ACTION_LEFT,
  ACTION_RIGHT,
  ACTION_STRAIGHT,
  DIR,
  SnakeEnv,
  type Direction,
} from "./env";

export const FEATURE_LABELS = [
  "danger ahead",
  "danger right",
  "danger left",
  "going right",
  "going down",
  "going left",
  "going up",
  "food left",
  "food right",
  "food up",
  "food down",
] as const;

export const ACTION_LABELS = ["straight", "right", "left"] as const;

export function extractState(env: SnakeEnv): Float64Array {
  const state = new Float64Array(11);
  const head = env.snake[0];
  const direction = env.direction;

  const actions = [ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT];
  for (let i = 0; i < 3; i++) {
    const nxtDir = SnakeEnv.turn(direction, actions[i]);
    const nxtPos = SnakeEnv.nextCell(head, nxtDir);
    state[i] = env.isCollision(nxtPos) ? 1 : 0;
  }

  state[3] = direction === DIR.RIGHT ? 1 : 0;
  state[4] = direction === DIR.DOWN ? 1 : 0;
  state[5] = direction === DIR.LEFT ? 1 : 0;
  state[6] = direction === DIR.UP ? 1 : 0;

  const { x: fx, y: fy } = env.food;
  state[7] = fx < head.x ? 1 : 0;
  state[8] = fx > head.x ? 1 : 0;
  state[9] = fy < head.y ? 1 : 0;
  state[10] = fy > head.y ? 1 : 0;

  return state;
}

export function directionName(d: Direction): string {
  return (["RIGHT", "DOWN", "LEFT", "UP"] as const)[d];
}
