"""
src/covid_env.py — COVID-era evaluation environment (Jan 2020 – Dec 2022).

Two modes:
  replay          — steps through real historical data month-by-month; agent
                    actions don't affect the trajectory, useful for visualizing
                    what policy the agent would have recommended.
  counterfactual  — starts MacroSimulator at Jan 2020 conditions and injects a
                    calibrated two-phase COVID shock (default, recommended for
                    meaningful policy comparison).

Historical data sourced from BLS (CPI-U, U-3) and FRED (EFFR).
"""

import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
import config
from fed_env import MacroSimulator

# Reward constants (mirror fed_env.py)
_REWARD_CLIP            = config.REWARD_CLIP
_RATE_VOLATILITY_WEIGHT = config.RATE_VOLATILITY_WEIGHT
_SOFT_LANDING_WEIGHT    = config.SOFT_LANDING_WEIGHT
_SOFT_LANDING_SIGMA     = config.SOFT_LANDING_SIGMA

# Real historical data (Jan 2020 – Dec 2025, index 0 = Jan 2020)
COVID_DATA: dict[str, list[float]] = {
    # BLS CPI-U, year-over-year %
    "cpi": [
        2.3, 2.3, 1.5,  0.3,  0.1,  0.6,   # Jan–Jun 2020
        1.0, 1.3, 1.4,  1.2,  1.2,  1.4,   # Jul–Dec 2020
        1.4, 1.7, 2.6,  4.2,  5.0,  5.4,   # Jan–Jun 2021
        5.4, 5.3, 5.4,  6.2,  6.8,  7.0,   # Jul–Dec 2021
        7.5, 7.9, 8.5,  8.3,  8.6,  9.1,   # Jan–Jun 2022
        8.5, 8.3, 8.2,  7.7,  7.1,  6.5,   # Jul–Dec 2022
        6.4, 6.0, 5.0,  4.9,  4.0,  3.0,   # Jan–Jun 2023
        3.2, 3.7, 3.7,  3.2,  3.1,  3.4,   # Jul–Dec 2023
        3.1, 3.2, 3.5,  3.4,  3.3,  3.0,   # Jan–Jun 2024
        2.9, 2.5, 2.4,  2.6,  2.7,  2.9,   # Jul–Dec 2024
        3.0, 2.8, 2.4,  2.3,  2.4,  2.7,   # Jan–Jun 2025
        2.9, 2.8, 2.5,  2.5,  2.5,  2.5,   # Jul–Dec 2025
    ],
    # BLS U-3 unemployment rate %
    "unemployment": [
         3.6,  3.5,  4.4, 14.7, 13.3, 11.1,  # Jan–Jun 2020
        10.2,  8.4,  7.9,  6.9,  6.7,  6.7,  # Jul–Dec 2020
         6.4,  6.2,  6.0,  6.0,  5.8,  5.9,  # Jan–Jun 2021
         5.4,  5.2,  4.8,  4.6,  4.2,  3.9,  # Jul–Dec 2021
         4.0,  3.8,  3.6,  3.6,  3.6,  3.6,  # Jan–Jun 2022
         3.5,  3.7,  3.5,  3.7,  3.6,  3.5,  # Jul–Dec 2022
         3.4,  3.6,  3.5,  3.4,  3.7,  3.6,  # Jan–Jun 2023
         3.5,  3.8,  3.8,  3.9,  3.7,  3.7,  # Jul–Dec 2023
         3.7,  3.9,  3.8,  3.9,  4.0,  4.1,  # Jan–Jun 2024
         4.3,  4.2,  4.1,  4.1,  4.2,  4.2,  # Jul–Dec 2024
         4.0,  4.1,  4.2,  4.2,  4.2,  4.1,  # Jan–Jun 2025
         4.2,  4.2,  4.2,  4.2,  4.2,  4.2,  # Jul–Dec 2025
    ],
    # FRED EFFR, monthly average %
    "effr": [
        1.75, 1.58, 0.65, 0.05, 0.05, 0.08,  # Jan–Jun 2020
        0.09, 0.10, 0.09, 0.09, 0.09, 0.09,  # Jul–Dec 2020
        0.08, 0.08, 0.07, 0.07, 0.05, 0.08,  # Jan–Jun 2021
        0.10, 0.09, 0.08, 0.08, 0.08, 0.08,  # Jul–Dec 2021
        0.08, 0.08, 0.20, 0.33, 0.77, 1.21,  # Jan–Jun 2022
        1.68, 2.33, 2.56, 3.08, 3.78, 4.10,  # Jul–Dec 2022
        4.33, 4.57, 4.65, 4.83, 5.06, 5.08,  # Jan–Jun 2023
        5.12, 5.33, 5.33, 5.33, 5.33, 5.33,  # Jul–Dec 2023
        5.33, 5.33, 5.33, 5.33, 5.33, 5.33,  # Jan–Jun 2024
        5.33, 5.33, 4.83, 4.83, 4.58, 4.33,  # Jul–Dec 2024  (Sep/Nov/Dec cuts)
        4.33, 4.33, 4.33, 4.33, 4.33, 4.33,  # Jan–Jun 2025  (on hold)
        4.33, 4.33, 4.33, 4.33, 4.33, 4.33,  # Jul–Dec 2025
    ],
}

_N_REAL = len(COVID_DATA["cpi"])   # 72 — used to clamp t_idx lookups
assert all(len(v) == _N_REAL for v in COVID_DATA.values()), \
    "COVID_DATA arrays must all have the same length"

# Initial conditions (Jan 2020)
INIT_PI   = 2.3    # CPI YoY %
INIT_U    = 3.6    # unemployment %
INIT_RATE = 1.75   # EFFR %

# Two-phase COVID shock schedule (counterfactual mode)
# Phase 1 — Lockdown (Apr–Jul 2020): unemployment spike, mild deflation
# Phase 2 — Supply chain (Mar 2021–Oct 2022): persistent inflation surge

_PHASE1_START   = 2
_PHASE1_END     = 5
_PHASE1_U_PUSH  =  3.5   # sharp unemployment spike
_PHASE1_PI_PUSH = -0.3   # mild deflationary pressure

_PHASE2_START   = 14
_PHASE2_END     = 34
_PHASE2_U_PUSH  =  0.2   # modest unemployment pressure
_PHASE2_PI_PUSH =  2.0   # strong inflation push


class CovidEnv(gym.Env):
    """
    RL evaluation environment calibrated to the COVID-era economy.

    Parameters
    ----------
    mode     : "counterfactual" (default) or "replay"
    llm_dim  : dimension of the llm_belief observation component
    """

    def __init__(self, mode: str = "counterfactual", llm_dim: int = config.LLM_DIM):
        super().__init__()
        if mode not in ("replay", "counterfactual"):
            raise ValueError(f"mode must be 'replay' or 'counterfactual', got {mode!r}")
        self.mode    = mode
        self.llm_dim = llm_dim
        self.sim     = MacroSimulator()

        self.action_mapping = {
            0: -0.75, 1: -0.50, 2: -0.25,
            3:  0.00,
            4:  0.25, 5:  0.50, 6:  0.75,
        }
        self.action_space = spaces.Discrete(len(self.action_mapping))

        self.observation_space = spaces.Dict({
            "macro": spaces.Box(
                low=np.array([-10.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([40.0, 30.0, 20.0], dtype=np.float32),
            ),
            "llm_belief": spaces.Box(
                low=-1.0, high=1.0, shape=(self.llm_dim,), dtype=np.float32
            ),
        })

        # Expose shock window for MockLLMObservationWrapper compatibility.
        # We expose the dominant (Phase 2) inflation shock as the primary shock.
        self.shock_start = _PHASE2_START
        self.shock_end   = _PHASE2_END
        self.shock_scale = 1.0

        self.t            = 0
        self.max_steps    = 120
        self.current_rate = INIT_RATE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t            = 0
        self.current_rate = INIT_RATE

        if self.mode == "counterfactual":
            self.sim.np_random = self.np_random
            self.sim.pi   = INIT_PI
            self.sim.u    = INIT_U
            self.sim.pi_e = INIT_PI

        return self._get_obs(), {}

    def step(self, action_idx: int):
        self.t += 1
        delta_rate        = self.action_mapping[int(action_idx)]
        self.current_rate = float(np.clip(self.current_rate + delta_rate, 0.0, 20.0))

        if self.mode == "replay":
            return self._step_replay(delta_rate)
        return self._step_counterfactual(delta_rate)

    def _step_replay(self, delta_rate: float):
        """Return the next real data row. Agent action affects rate tracking
        and the volatility component of the reward, but not the trajectory."""
        t_idx    = min(self.t, _N_REAL - 1)
        real_pi  = COVID_DATA["cpi"][t_idx]
        real_u   = COVID_DATA["unemployment"][t_idx]
        real_rate = COVID_DATA["effr"][t_idx]

        reward   = self._compute_reward(real_pi, real_u, delta_rate)
        phase    = self._get_phase()
        regime   = "normal" if phase is None else f"phase{phase}"
        info     = {
            "regime":    regime,
            "phase":     phase,
            "real_pi":   real_pi,
            "real_u":    real_u,
            "real_rate": real_rate,
        }
        truncated = bool(self.t >= self.max_steps)
        return self._get_obs(), float(reward), False, truncated, info

    def _step_counterfactual(self, delta_rate: float):
        """Advance MacroSimulator with the phase-appropriate COVID shock."""
        phase = self._get_phase()

        if phase == 1:
            self._sim_step_custom(
                self.current_rate,
                u_push=_PHASE1_U_PUSH,
                pi_push=_PHASE1_PI_PUSH,
                apply_rate_floor=True,
            )
        elif phase == 2:
            self._sim_step_custom(
                self.current_rate,
                u_push=_PHASE2_U_PUSH,
                pi_push=_PHASE2_PI_PUSH,
                apply_rate_floor=True,
            )
        else:
            self.sim.step(self.current_rate, shock_regime="normal")

        t_idx     = min(self.t, _N_REAL - 1)
        real_pi   = COVID_DATA["cpi"][t_idx]
        real_u    = COVID_DATA["unemployment"][t_idx]
        real_rate = COVID_DATA["effr"][t_idx]

        reward  = self._compute_reward(self.sim.pi, self.sim.u, delta_rate)
        regime  = "normal" if phase is None else f"phase{phase}"
        info    = {
            "regime":    regime,
            "phase":     phase,
            "real_pi":   real_pi,
            "real_u":    real_u,
            "real_rate": real_rate,
        }
        truncated = bool(self.t >= self.max_steps)
        return self._get_obs(), float(reward), False, truncated, info

    def _sim_step_custom(
        self,
        nominal_rate: float,
        u_push: float,
        pi_push: float,
        apply_rate_floor: bool = True,
    ):
        """One MacroSimulator step with explicit phase-specific shock pushes."""
        sim = self.sim
        real_rate = nominal_rate - sim.pi_e
        rate_gap  = real_rate - sim.r_star

        shock_u  = sim.np_random.normal(0, 0.1) + u_push
        shock_pi = sim.np_random.normal(0, 0.1) + pi_push

        # Supply-constraint floor: cap the stimulative effect of loose monetary
        # policy during a supply shock (same logic as MacroSimulator "supply" regime)
        if apply_rate_floor:
            rate_gap = max(rate_gap, 0.0)

        # IS curve / unemployment dynamics
        u_gap  = sim.u - sim.u_star
        new_u  = sim.u_star + (sim.rho_u * u_gap) + (sim.alpha * rate_gap) + shock_u
        sim.u  = float(np.clip(new_u, 1.0, 25.0))

        # Phillips curve
        pi_gap     = sim.pi - sim.pi_star
        new_u_gap  = sim.u  - sim.u_star
        new_pi     = sim.pi_star + (sim.rho_pi * pi_gap) - (sim.kappa * new_u_gap) + shock_pi
        sim.pi     = float(np.clip(new_pi, -5.0, 30.0))

        # Adaptive inflation expectations
        sim.pi_e = 0.5 * sim.pi_e + 0.5 * sim.pi

    def _compute_reward(self, pi: float, u: float, delta_rate: float) -> float:
        pi_loss          = (pi - self.sim.pi_star) ** 2
        u_loss           = (u  - self.sim.u_star)  ** 2
        u_fear_penalty   = 5.0 * (u - 6.0) ** 2 if u > 6.0 else 0.0
        rate_vol_loss    = _RATE_VOLATILITY_WEIGHT * delta_rate ** 2

        pi_gap = pi - self.sim.pi_star
        u_gap  = u  - self.sim.u_star
        soft_landing_bonus = _SOFT_LANDING_WEIGHT * float(np.exp(
            -0.5 * ((pi_gap / _SOFT_LANDING_SIGMA) ** 2
                    + (u_gap  / _SOFT_LANDING_SIGMA) ** 2)
        ))

        reward = -(pi_loss + u_loss + u_fear_penalty + rate_vol_loss) + soft_landing_bonus
        return max(reward, _REWARD_CLIP)

    def _get_obs(self):
        t_idx = min(self.t, _N_REAL - 1)
        if self.mode == "replay":
            pi   = COVID_DATA["cpi"][t_idx]
            u    = COVID_DATA["unemployment"][t_idx]
            rate = COVID_DATA["effr"][t_idx]
        else:
            pi   = self.sim.pi
            u    = self.sim.u
            rate = self.current_rate

        return {
            "macro":      np.array([pi, u, rate], dtype=np.float32),
            "llm_belief": np.zeros(self.llm_dim, dtype=np.float32),
        }

    def _get_phase(self) -> int | None:
        """Return the active COVID phase (1, 2) or None for inter-shock months."""
        if _PHASE1_START <= self.t <= _PHASE1_END:
            return 1
        if _PHASE2_START <= self.t <= _PHASE2_END:
            return 2
        return None

    @property
    def real_trajectory(self) -> dict[str, list[float]]:
        """Full COVID historical data dict — convenient for overlay plotting."""
        return COVID_DATA


def _taylor_action(obs: dict, action_mapping: dict) -> int:
    """Standard Taylor Rule (same implementation as benchmark.py)."""
    pi, u, current_rate = obs["macro"]
    target_rate   = 2.0 + pi + 0.5 * (pi - 2.0) - 0.5 * (u - 4.0)
    desired_delta = target_rate - current_rate
    best_action, min_diff = 3, float("inf")
    for idx, delta in action_mapping.items():
        diff = abs(delta - desired_delta)
        if diff < min_diff:
            min_diff = diff
            best_action = idx
    return best_action


def covid_eval(
    model=None,
    env_factory=None,
    mode: str = "counterfactual",
    n_runs: int = 1,
    policy: str = "mlp",
    seed: int = 0,
) -> dict:
    """
    Evaluate a policy on the COVID-era environment.

    Parameters
    ----------
    model       : PPO / RecurrentPPO model, or None → Taylor Rule
    env_factory : callable returning a (wrapped) CovidEnv.
                  Defaults to ``lambda: CovidEnv(mode=mode)``.
    mode        : "counterfactual" (default) or "replay"
    n_runs      : number of independent runs (averaged over different seeds)
    policy      : "mlp" or "lstm" — how model.predict() is called
    seed        : base RNG seed; run i uses seed+i

    Returns
    -------
    dict with keys:
      "rewards"         : list[float] — per-run episode rewards
      "trajectories"    : list[dict]  — per-run {pi, u, rate, phase, ep_reward}
      "real_trajectory" : dict        — COVID_DATA for overlay plotting
      "mean_reward"     : float
      "std_reward"      : float
    """
    if env_factory is None:
        _mode = mode
        env_factory = lambda: CovidEnv(mode=_mode)

    rewards: list[float] = []
    trajectories: list[dict] = []
    real_traj: dict | None = None

    for run_i in range(n_runs):
        env = env_factory()
        obs, _ = env.reset(seed=seed + run_i)
        real_traj = env.unwrapped.real_trajectory

        ep_reward     = 0.0
        done          = False
        lstm_states   = None
        episode_start = np.ones((1,), dtype=bool)
        pi_hist, u_hist, rate_hist, phase_hist = [], [], [], []

        while not done:
            pi, u, rate = obs["macro"]
            pi_hist.append(float(pi))
            u_hist.append(float(u))
            rate_hist.append(float(rate))

            if model is None:
                action = _taylor_action(obs, env.unwrapped.action_mapping)
            elif policy == "lstm":
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action = int(action)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            phase_hist.append(info.get("phase"))
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        trajectories.append({
            "pi":       pi_hist,
            "u":        u_hist,
            "rate":     rate_hist,
            "phase":    phase_hist,
            "ep_reward": ep_reward,
        })
        env.close()

    return {
        "rewards":          rewards,
        "trajectories":     trajectories,
        "real_trajectory":  real_traj,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
    }
