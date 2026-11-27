"""Small JAX numerical tests; no checkpoint/GPU/hardware required."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from numpy.testing import assert_allclose
from openpi.models.rtc_sampler import guided_velocity


def test_zero_weights_preserve_original_velocity():
    x = jnp.array([0.2, -0.3])
    velocity = lambda z: jnp.sin(z)
    result = jax.jit(guided_velocity, static_argnums=(0,))(
        velocity, x, 0.6, jnp.ones(2), jnp.zeros(2), 5.0
    )
    assert_allclose(result, velocity(x))


def test_guidance_decreases_target_error_with_negative_dt():
    x = jnp.zeros(3)
    target = jnp.array([0.2, -0.2, 0.0])
    velocity = lambda z: z * 0
    v = guided_velocity(velocity, x, 1.0, target, jnp.ones(3), 5.0)
    updated = x - 0.1 * v
    assert np.linalg.norm(updated - target) < np.linalg.norm(x - target)
    assert_allclose(updated, 0.5 * target)


def test_denoiser_vjp_matches_analytic_nonlinear_gradient():
    x = jnp.array([0.2, 0.3])
    t = 0.4
    target = jnp.array([0.7, -0.1])
    weights = jnp.array([1.0, 0.3])
    velocity = lambda z: z * z
    estimate = x - t * x * x
    coefficient = min(5.0, t / ((1 - t) * (t * t / ((1 - t) ** 2 + t * t))))
    expected = x * x - coefficient * (1 - 2 * t * x) * (target - estimate) * weights
    assert_allclose(
        guided_velocity(velocity, x, t, target, weights, 5.0), expected, rtol=1e-6
    )


def test_full_sampler_zero_guidance_matches_euler(monkeypatch):
    """Run the new sampler on a toy velocity model, not on a real checkpoint."""
    import sys
    from types import ModuleType, SimpleNamespace
    from functools import partial
    from openpi.models.rtc_sampler import sample_actions_rtc

    model_module = ModuleType("openpi.models.model")
    model_module.preprocess_observation = lambda _, obs, train: obs
    pi0_module = ModuleType("openpi.models.pi0")
    pi0_module.make_attn_mask = lambda mask, ar: mask[:, :, None] & mask[:, None, :]
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    monkeypatch.setitem(sys.modules, "openpi.models.pi0", pi0_module)
    import openpi.models

    monkeypatch.setattr(openpi.models, "model", model_module, raising=False)
    monkeypatch.setattr(openpi.models, "pi0", pi0_module, raising=False)

    class ToyModel:
        action_horizon = 4
        action_dim = 3

        def __init__(self):
            self.PaliGemma = SimpleNamespace(llm=self.llm)

        def llm(self, tokens, **kwargs):
            return tokens, None

        def embed_prefix(self, obs):
            return jnp.zeros((1, 2, 3)), jnp.ones((1, 2), bool), jnp.zeros(2, bool)

        def embed_suffix(self, obs, z, t):
            return (
                z * 0.2 + t[:, None, None] * 0.1,
                jnp.ones((1, 4), bool),
                jnp.ones(4, bool),
                None,
            )

        def action_out_proj(self, z):
            return z

    model = ToyModel()
    # Observation is closed over here; only the action arrays are JIT inputs.
    obs = SimpleNamespace(state=jnp.zeros((1, 3)))
    sample = jax.jit(partial(sample_actions_rtc, model, jax.random.key(0), obs))
    noise = jnp.ones((1, 4, 3)) * 0.7
    result = sample(
        prefix=jnp.zeros_like(noise),
        weights=jnp.zeros_like(noise),
        noise=noise,
        num_steps=10,
    )
    expected = np.asarray(noise)
    for i in range(10):
        expected = expected - (expected * 0.2 + (1 - i / 10) * 0.1) / 10
    assert_allclose(result, expected, atol=1e-6)
    guided = sample(
        prefix=jnp.zeros_like(noise),
        weights=jnp.ones_like(noise),
        noise=noise,
        num_steps=10,
    )
    assert np.linalg.norm(guided) < np.linalg.norm(result)


def test_nnx_binding_preserves_model_parameters(monkeypatch):
    """Exercise the real OpenPI NNX wrapper with a small parameterized module."""
    from types import ModuleType, MethodType
    from typing import NamedTuple
    import sys
    from flax import nnx
    import openpi.models
    from openpi.shared.nnx_utils import module_jit
    from openpi.models.rtc_sampler import sample_actions_rtc

    model_module = ModuleType("openpi.models.model")
    model_module.preprocess_observation = lambda _, obs, train: obs
    pi0_module = ModuleType("openpi.models.pi0")
    pi0_module.make_attn_mask = lambda mask, ar: mask[:, :, None] & mask[:, None, :]
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    monkeypatch.setitem(sys.modules, "openpi.models.pi0", pi0_module)
    monkeypatch.setattr(openpi.models, "model", model_module, raising=False)
    monkeypatch.setattr(openpi.models, "pi0", pi0_module, raising=False)

    class Observation(NamedTuple):
        state: object

    class LLM(nnx.Module):
        def __call__(self, tokens, **kwargs):
            return tokens, None

    class Toy(nnx.Module):
        def __init__(self):
            self.action_horizon = 4
            self.action_dim = 3
            self.gain = nnx.Param(jnp.asarray(0.2))
            self.PaliGemma = nnx.Dict(llm=LLM())

        def embed_prefix(self, obs):
            return jnp.zeros((1, 2, 3)), jnp.ones((1, 2), bool), jnp.zeros(2, bool)

        def embed_suffix(self, obs, z, t):
            return z * self.gain.value, jnp.ones((1, 4), bool), jnp.ones(4, bool), None

        def action_out_proj(self, z):
            return z

    model = Toy()
    sampler = module_jit(MethodType(sample_actions_rtc, model))
    obs = Observation(jnp.zeros((1, 3)))
    noise = jnp.ones((1, 4, 3)) * 0.7
    kwargs = dict(
        prefix=jnp.zeros_like(noise), noise=noise, num_steps=10, max_guidance=5.0
    )
    ordinary = sampler(jax.random.key(0), obs, weights=jnp.zeros_like(noise), **kwargs)
    guided = sampler(jax.random.key(0), obs, weights=jnp.ones_like(noise), **kwargs)
    assert np.linalg.norm(guided) < np.linalg.norm(ordinary)
    assert_allclose(model.gain.value, 0.2)
