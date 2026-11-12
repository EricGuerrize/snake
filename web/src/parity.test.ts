import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { MLPQNetwork, type ModelWeights } from "./network";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

describe("paridade JS ↔ Python (forward)", () => {
  it("Q-values e ações batem com a fixture exportada", () => {
    const model = JSON.parse(
      readFileSync(join(root, "public/model.json"), "utf8"),
    ) as ModelWeights;
    const fixture = JSON.parse(
      readFileSync(join(root, "public/parity_fixture.json"), "utf8"),
    ) as {
      atol: number;
      samples: { state: number[]; q: number[]; action: number }[];
    };

    const net = new MLPQNetwork(model);
    for (const sample of fixture.samples) {
      const { q, action } = net.forward(sample.state);
      expect(action).toBe(sample.action);
      for (let i = 0; i < 3; i++) {
        expect(Math.abs(q[i] - sample.q[i])).toBeLessThan(fixture.atol * 10 + 1e-8);
      }
    }
  });
});
