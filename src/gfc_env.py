"""
src/gfc_env.py — 2008 Global Financial Crisis evaluation environment (Jan 2007 – Dec 2012).

Two modes:
  replay          — steps through real historical data month-by-month; agent
                    actions don't affect the trajectory, useful for visualizing
                    what policy the agent would have recommended.
  counterfactual  — starts MacroSimulator at Jan 2007 conditions and injects a
                    calibrated three-phase GFC shock (default, recommended for
                    meaningful policy comparison).

Historical data sourced from BLS (CPI-U, U-3) and FRED (EFFR).

Three-phase shock structure:
  Phase 1 — Housing Bust + Commodity Inflation (Jul 2007 – Aug 2008):
      Gradual unemployment rise from credit tightening; elevated inflation
      driven by oil/commodity prices. The Fed faces a stagflationary dilemma
      (rising unemployment but inflation well above target).
  Phase 2 — Acute Financial Crisis (Sep 2008 – Mar 2009):
      Lehman collapse; credit markets freeze; sharp unemployment surge;
      demand collapses producing rapid disinflation/deflation. Policy
      transmission is impaired (rate floor models the broken credit channel).
  Phase 3 — Extended Recession / Jobless Recovery (Apr 2009 – Dec 2010):
      Unemployment stabilises at historically high levels; persistent
      below-target inflation; conventional monetary policy exhausted (ZLB).
"""

import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
import config
from fed_env import MacroSimulator

# Reward constants (mirror fed_env.py / covid_env.py)
_REWARD_CLIP            = config.REWARD_CLIP
_RATE_VOLATILITY_WEIGHT = config.RATE_VOLATILITY_WEIGHT
_SOFT_LANDING_WEIGHT    = config.SOFT_LANDING_WEIGHT
_SOFT_LANDING_SIGMA     = config.SOFT_LANDING_SIGMA

# ---------------------------------------------------------------------------
# Real historical data  (Jan 2007 – Dec 2012, index 0 = Jan 2007)
# ---------------------------------------------------------------------------
GFC_DATA: dict[str, list[float]] = {
    # BLS CPI-U, year-over-year %
    "cpi": [
        # 2007
        2.1, 2.4, 2.8, 2.6, 2.7, 2.7, 2.4, 2.0, 2.8, 3.5, 4.3, 4.1,
        # 2008  (commodity spike then crash)
        4.3, 4.0, 4.0, 3.9, 4.2, 5.0, 5.6, 5.4, 4.9, 3.7, 1.1, 0.1,
        # 2009  (deflation trough)
        0.0, 0.2, -0.4, -0.7, -1.3, -1.4, -2.1, -1.5, -1.3, -0.2, 1.8, 2.7,
        # 2010  (recovery, still below target)
        2.6, 2.1, 2.3, 2.2, 2.0, 1.1, 1.2, 1.1, 1.1, 1.2, 1.1, 1.5,
        # 2011  (commodity reflation)
        1.6, 2.1, 2.7, 3.2, 3.6, 3.6, 3.6, 3.8, 3.9, 3.5, 3.4, 3.0,
        # 2012  (gradual normalisation)
        2.9, 2.9, 2.7, 2.3, 1.7, 1.7, 1.4, 1.7, 2.0, 2.2, 1.8, 1.7,
    ],
    # BLS U-3 unemployment rate %
    "unemployment": [
        # 2007
        4.6, 4.5, 4.4, 4.5, 4.4, 4.6, 4.7, 4.6, 4.7, 4.7, 4.7, 5.0,
        # 2008
        5.0, 4.9, 5.1, 5.0, 5.4, 5.6, 5.8, 6.1, 6.1, 6.5, 6.8, 7.3,
        # 2009
        7.8, 8.3, 8.7, 9.0, 9.4, 9.5, 9.5, 9.6, 9.8, 10.0, 9.9, 9.9,
        # 2010
        9.7, 9.8, 9.8, 9.9, 9.6, 9.5, 9.5, 9.5, 9.6, 9.5, 9.8, 9.4,
        # 2011
        9.1, 9.0, 8.9, 9.0, 9.1, 9.2, 9.1, 9.1, 9.1, 8.9, 8.7, 8.5,
        # 2012
        8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.3, 8.1, 7.8, 7.9, 7.8, 7.9,
    ],
    # FRED EFFR, monthly average %
    "effr": [
        # 2007  (gradual cuts begin in Sep)
        5.25, 5.26, 5.26, 5.25, 5.25, 5.25, 5.26, 5.02, 4.94, 4.76, 4.65, 4.24,
        # 2008  (aggressive cuts; near-zero by Dec)
        3.94, 3.50, 2.98, 2.48, 2.00, 2.00, 2.00, 2.00, 1.81, 1.00, 0.39, 0.16,
        # 2009–2012  (zero-lower-bound era)
        0.15, 0.22, 0.18, 0.15, 0.18, 0.21, 0.16, 0.16, 0.15, 0.12, 0.12, 0.12,
        0.11, 0.13, 0.16, 0.20, 0.20, 0.18, 0.18, 0.19, 0.19, 0.19, 0.19, 0.20,
        0.17, 0.16, 0.14, 0.10, 0.09, 0.09, 0.07, 0.10, 0.08, 0.07, 0.08, 0.07,
        0.08, 0.10, 0.13, 0.14, 0.18, 0.18, 0.16, 0.14, 0.14, 0.16, 0.16, 0.16,
    ],
}

_N_REAL = len(GFC_DATA["cpi"])   # 72
assert all(len(v) == _N_REAL for v in GFC_DATA.values()), \
    "GFC_DATA arrays must all have the same length"

# Initial conditions (Jan 2007)
INIT_PI   = 2.1    # CPI YoY %
INIT_U    = 4.6    # unemployment %
INIT_RATE = 5.25   # EFFR %

# ---------------------------------------------------------------------------
# Three-phase GFC shock schedule (counterfactual mode)
# ---------------------------------------------------------------------------

# Phase 1 — Housing Bust + Commodity Inflation (Jul 2007 – Aug 2008)
# Credit tightening from subprime collapse slowly pushes up unemployment while
# oil/commodity prices drive inflation well above target.
_PHASE1_START   = 6    # Jul 2007
_PHASE1_END     = 19   # Aug 2008
_PHASE1_U_PUSH  =  0.25  # gradual unemployment rise
_PHASE1_PI_PUSH =  0.8   # commodity-driven inflation surge

# Phase 2 — Acute Financial Crisis (Sep 2008 – Mar 2009)
# Lehman collapse; credit freeze; demand collapses; rapid disinflation.
# Rate floor models the broken credit/lending channel.
_PHASE2_START   = 20   # Sep 2008
_PHASE2_END     = 26   # Mar 2009
_PHASE2_U_PUSH  =  1.5   # sharp unemployment spike
_PHASE2_PI_PUSH = -2.5   # deflationary demand collapse

# Phase 3 — Extended Recession / Jobless Recovery (Apr 2009 – Dec 2010)
# Unemployment entrenched at high levels; below-target inflation; ZLB binds.
_PHASE3_START   = 27   # Apr 2009
_PHASE3_END     = 47   # Dec 2010
_PHASE3_U_PUSH  =  0.1   # unemployment plateaus at elevated level
_PHASE3_PI_PUSH = -0.5   # persistent low/negative inflation


class GFCEnv(gym.Env):
    """
    RL evaluation environment calibrated to the 2008 Global Financial Crisis.

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

        # Expose the dominant acute-crisis shock window for MockLLMObservationWrapper
        # compatibility (mirrors the convention in covid_env.py).
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

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------

    def _step_replay(self, delta_rate: float):
        """Return the next real data row. Agent action affects rate tracking
        and the volatility component of the reward, but not the trajectory."""
        t_idx     = min(self.t, _N_REAL - 1)
        real_pi   = GFC_DATA["cpi"][t_idx]
        real_u    = GFC_DATA["unemployment"][t_idx]
        real_rate = GFC_DATA["effr"][t_idx]

        reward  = self._compute_reward(real_pi, real_u, delta_rate)
        phase   = self._get_phase()
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

    # ------------------------------------------------------------------
    # Counterfactual mode
    # ------------------------------------------------------------------

    def _step_counterfactual(self, delta_rate: float):
        """Advance MacroSimulator with the phase-appropriate GFC shock."""
        phase = self._get_phase()

        if phase == 1:
            # Stagflationary dilemma: unemployment rises but commodity inflation
            # is above target → no supply-constraint floor (standard IS/PC dynamics)
            self._sim_step_custom(
                self.current_rate,
                u_push=_PHASE1_U_PUSH,
                pi_push=_PHASE1_PI_PUSH,
                apply_rate_floor=False,
            )
        elif phase == 2:
            # Credit channel broken: apply rate floor to model impaired
            # policy transmission during acute financial panic
            self._sim_step_custom(
                self.current_rate,
                u_push=_PHASE2_U_PUSH,
                pi_push=_PHASE2_PI_PUSH,
                apply_rate_floor=True,
            )
        elif phase == 3:
            # ZLB era: rate floor models entrenched slack and limited
            # conventional monetary policy effectiveness
            self._sim_step_custom(
                self.current_rate,
                u_push=_PHASE3_U_PUSH,
                pi_push=_PHASE3_PI_PUSH,
                apply_rate_floor=True,
            )
        else:
            self.sim.step(self.current_rate, shock_regime="normal")

        t_idx     = min(self.t, _N_REAL - 1)
        real_pi   = GFC_DATA["cpi"][t_idx]
        real_u    = GFC_DATA["unemployment"][t_idx]
        real_rate = GFC_DATA["effr"][t_idx]

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sim_step_custom(
        self,
        nominal_rate: float,
        u_push: float,
        pi_push: float,
        apply_rate_floor: bool = False,
    ):
        """One MacroSimulator step with explicit phase-specific shock pushes."""
        sim = self.sim
        real_rate = nominal_rate - sim.pi_e
        rate_gap  = real_rate - sim.r_star

        shock_u  = u_push
        shock_pi = pi_push

        # When the credit/transmission channel is impaired (Phase 2/3),
        # cap the stimulative effect of loose monetary policy.
        if apply_rate_floor:
            rate_gap = max(rate_gap, 0.0)

        # IS curve / unemployment dynamics
        u_gap  = sim.u - sim.u_star
        new_u  = sim.u_star + (sim.rho_u * u_gap) + (sim.alpha * rate_gap) + shock_u
        sim.u  = float(np.clip(new_u, 1.0, 25.0))

        # Phillips curve
        pi_gap    = sim.pi - sim.pi_star
        new_u_gap = sim.u  - sim.u_star
        new_pi    = sim.pi_star + (sim.rho_pi * pi_gap) - (sim.kappa * new_u_gap) + shock_pi
        sim.pi    = float(np.clip(new_pi, -5.0, 30.0))

        # Adaptive inflation expectations
        sim.pi_e = 0.5 * sim.pi_e + 0.5 * sim.pi

    def _compute_reward(self, pi: float, u: float, delta_rate: float) -> float:
        pi_loss        = (pi - self.sim.pi_star) ** 2
        u_loss         = (u  - self.sim.u_star)  ** 2
        u_fear_penalty = 5.0 * (u - 6.0) ** 2 if u > 6.0 else 0.0
        rate_vol_loss  = _RATE_VOLATILITY_WEIGHT * delta_rate ** 2

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
            pi   = GFC_DATA["cpi"][t_idx]
            u    = GFC_DATA["unemployment"][t_idx]
            rate = GFC_DATA["effr"][t_idx]
        else:
            pi   = self.sim.pi
            u    = self.sim.u
            rate = self.current_rate

        return {
            "macro":      np.array([pi, u, rate], dtype=np.float32),
            "llm_belief": np.zeros(self.llm_dim, dtype=np.float32),
        }

    def _get_phase(self) -> int | None:
        """Return the active GFC phase (1, 2, 3) or None for inter-shock months."""
        if _PHASE1_START <= self.t <= _PHASE1_END:
            return 1
        if _PHASE2_START <= self.t <= _PHASE2_END:
            return 2
        if _PHASE3_START <= self.t <= _PHASE3_END:
            return 3
        return None

    @property
    def real_trajectory(self) -> dict[str, list[float]]:
        """Full GFC historical data dict — convenient for overlay plotting."""
        return GFC_DATA


# ---------------------------------------------------------------------------
# Standalone Taylor Rule helper (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def _taylor_action(obs: dict, action_mapping: dict) -> int:
    """Standard Taylor Rule."""
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


# ---------------------------------------------------------------------------
# Evaluation helper (mirrors covid_eval)
# ---------------------------------------------------------------------------

def gfc_eval(
    model=None,
    env_factory=None,
    mode: str = "counterfactual",
    n_runs: int = 1,
    policy: str = "mlp",
    seed: int = 0,
) -> dict:
    """
    Evaluate a policy on the GFC-era environment.

    Parameters
    ----------
    model       : PPO / RecurrentPPO model, or None → Taylor Rule
    env_factory : callable returning a (wrapped) GFCEnv.
                  Defaults to ``lambda: GFCEnv(mode=mode)``.
    mode        : "counterfactual" (default) or "replay"
    n_runs      : number of independent runs (averaged over different seeds)
    policy      : "mlp" or "lstm" — how model.predict() is called
    seed        : base RNG seed; run i uses seed+i

    Returns
    -------
    dict with keys:
      "rewards"         : list[float] — per-run episode rewards
      "trajectories"    : list[dict]  — per-run {pi, u, rate, phase, ep_reward}
      "real_trajectory" : dict        — GFC_DATA for overlay plotting
      "mean_reward"     : float
      "std_reward"      : float
    """
    if env_factory is None:
        _mode = mode
        env_factory = lambda: GFCEnv(mode=_mode)

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
            "pi":        pi_hist,
            "u":         u_hist,
            "rate":      rate_hist,
            "phase":     phase_hist,
            "ep_reward": ep_reward,
        })
        env.close()

    return {
        "rewards":         rewards,
        "trajectories":    trajectories,
        "real_trajectory": real_traj,
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
    }
