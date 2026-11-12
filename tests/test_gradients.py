"""Checagem numérica: backprop analítico ≈ diferenças finitas."""

from __future__ import annotations

import numpy as np
from snake_rl.agent import MLPQNetwork


def test_manual_backprop_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    net = MLPQNetwork(input_size=11, hidden_size=8, output_size=3, rng=rng)

    batch = 4
    x = rng.normal(size=(batch, 11))
    # Entradas tipicamente binárias — mistura para cobrir ReLU on/off.
    x = (x > 0).astype(np.float64)
    actions = rng.integers(0, 3, size=batch)
    targets = rng.normal(size=batch)

    rel_errors = net.numerical_gradient_check(x, actions, targets, eps=1e-5)

    for name, err in rel_errors.items():
        assert err < 1e-5, f"gradiente de {name} divergiu: rel_error={err}"


def test_forward_shapes() -> None:
    net = MLPQNetwork(rng=np.random.default_rng(1))
    q1 = net.forward(np.zeros(11))
    assert q1.shape == (3,)
    q2 = net.forward(np.zeros((5, 11)))
    assert q2.shape == (5, 3)


def test_save_load_roundtrip(tmp_path) -> None:
    from snake_rl.agent import DQNAgent

    agent = DQNAgent(seed=3)
    path = tmp_path / "w.npz"
    agent.save(path)

    other = DQNAgent(seed=99)
    other.load(path)
    x = np.ones(11)
    np.testing.assert_allclose(agent.online.forward(x), other.online.forward(x))
