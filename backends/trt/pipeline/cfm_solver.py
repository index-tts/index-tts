"""Standalone CFM (Conditional Flow Matching) Euler ODE solver."""

import torch


@torch.inference_mode()
def cfm_inference(
    estimator_fn,
    mu: torch.Tensor,
    x_lens: torch.Tensor,
    prompt: torch.Tensor,
    style: torch.Tensor,
    n_timesteps: int = 25,
    temperature: float = 1.0,
    inference_cfg_rate: float = 0.7,
    zero_prompt_speech_token: bool = False,
    prompt_lens: torch.Tensor = None,
) -> torch.Tensor:
    """Run CFM inference using a fixed Euler ODE solver.

    Args:
        estimator_fn: callable(x, prompt_x, x_lens, t, style, cond) -> velocity.
            Typically the DiT TRT engine's __call__.
        mu: (B, T, 512) semantic conditioning (cat_condition).
        x_lens: (B,) per-sample total mel lengths (prompt + generated).
        prompt: (B, 80, T_prompt_max) reference mel spectrogram (padded to max).
        style: (B, 192) global speaker style from CAMPPlus.
        n_timesteps: number of Euler steps (default 25).
        temperature: noise scaling (default 1.0).
        inference_cfg_rate: classifier-free guidance rate (default 0.7).
        zero_prompt_speech_token: zero out mu in the prompt region.
        prompt_lens: (B,) per-sample prompt lengths. If None, uses prompt.size(-1)
            for all samples (backward compatible).

    Returns:
        x: (B, 80, T) final mel spectrogram.
    """
    B, T = mu.size(0), mu.size(1)
    device = mu.device

    z = torch.randn([B, prompt.size(1), T], device=device) * temperature
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)

    t = t_span[0]

    if prompt_lens is None:
        prompt_lens = torch.full((B,), prompt.size(-1), dtype=torch.long, device=device)
    max_prompt_len = prompt_lens.max().item()

    # Per-sample prompt_x and zeroing of z/mu in prompt regions
    prompt_x = torch.zeros_like(z)
    for b in range(B):
        pl = prompt_lens[b].item()
        prompt_x[b, :, :pl] = prompt[b, :, :pl]
        z[b, :, :pl] = 0
        if zero_prompt_speech_token:
            mu[b, :, :pl] = 0

    x = z
    for step in range(1, len(t_span)):
        dt = t_span[step] - t_span[step - 1]

        t_batch = t.unsqueeze(0).expand(B)

        if inference_cfg_rate > 0:
            stacked_x = torch.cat([x, x], dim=0)
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_t = torch.cat([t_batch, t_batch], dim=0)
            stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)

            stacked_dphi_dt = estimator_fn(
                stacked_x, stacked_prompt_x, stacked_x_lens,
                stacked_t, stacked_style, stacked_mu,
            )

            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        else:
            dphi_dt = estimator_fn(x, prompt_x, x_lens, t_batch, style, mu)

        x = x + dt * dphi_dt
        t = t + dt
        # Zero prompt region per-sample
        for b in range(B):
            x[b, :, :prompt_lens[b].item()] = 0

    return x
