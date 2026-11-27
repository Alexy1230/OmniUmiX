"""Inference-only Pi0/Pi05 RTC guidance (arXiv:2506.07339).

No model parameters or existing sample_actions signature are changed. Bind
sample_actions_rtc to the loaded Pi0 instance with MethodType/module_jit.
OpenPI uses t=1 noise -> t=0 data, opposite the paper's tau=1-t.
"""

import einops
import jax
import jax.numpy as jnp


def guided_velocity(velocity, x, t, target, weights, max_guidance):
    """Eq. 2--4 in decreasing-time convention, including the denoiser VJP."""

    def denoise(z):
        v = velocity(z)
        return z - t * v, v

    estimate, pullback, v = jax.vjp(denoise, x, has_aux=True)
    error = (target - estimate) * weights
    gradient = pullback(error)[0]
    tau = 1.0 - t
    r_squared = t * t / (tau * tau + t * t)
    strength = jnp.minimum(max_guidance, t / jnp.maximum(tau * r_squared, 1e-8))
    # dt is negative: subtract guidance to move towards target on integration.
    return v - strength * gradient


def sample_actions_rtc(
    self,
    rng,
    observation,
    *,
    prefix,
    weights,
    num_steps=10,
    max_guidance=5.0,
    noise=None
):
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm(
        [tokens, None],
        mask=make_attn_mask(mask, ar_mask),
        positions=jnp.cumsum(mask, axis=1) - 1,
    )

    def step(index, x):
        t = 1.0 - index / num_steps

        def velocity(z):
            suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(
                observation, z, jnp.broadcast_to(t, batch_size)
            )
            suffix_attention = make_attn_mask(suffix_mask, suffix_ar)
            prefix_attention = einops.repeat(mask, "b p -> b s p", s=suffix.shape[1])
            attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
            positions = (
                jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, output), _ = self.PaliGemma.llm(
                [None, suffix],
                mask=attention,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, adarms],
            )
            return self.action_out_proj(output[:, -self.action_horizon :])

        v = guided_velocity(velocity, x, t, prefix, weights, max_guidance)
        return x - v / num_steps

    return jax.lax.fori_loop(0, num_steps, step, noise)
