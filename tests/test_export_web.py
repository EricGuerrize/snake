"""Testes do exportador para o front-end web."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from snake_rl.agent import DQNAgent
from snake_rl.export_web import export_model_json, export_parity_fixture, weights_to_json


def test_weights_to_json_roundtrip(tmp_path: Path) -> None:
    agent = DQNAgent(seed=0)
    npz = tmp_path / "m.npz"
    agent.save(npz)
    payload = weights_to_json(npz)
    assert payload["architecture"] == "11>64>3"
    assert len(payload["W1"]) == 11
    assert len(payload["W1"][0]) == 64
    assert len(payload["b2"]) == 3

    out = export_model_json(npz, tmp_path / "model.json")
    loaded = json.loads(out.read_text())
    np.testing.assert_allclose(loaded["W1"], agent.online.W1)


def test_parity_fixture_matches_python(tmp_path: Path) -> None:
    agent = DQNAgent(seed=1)
    npz = tmp_path / "m.npz"
    agent.save(npz)
    fixture_path = export_parity_fixture(npz, tmp_path / "parity.json", n_states=8, seed=3)
    fixture = json.loads(fixture_path.read_text())
    agent.load(npz)
    for sample in fixture["samples"]:
        q = agent.online.forward(np.asarray(sample["state"], dtype=np.float64))
        assert int(np.argmax(q)) == sample["action"]
        np.testing.assert_allclose(q, sample["q"], atol=1e-12)
