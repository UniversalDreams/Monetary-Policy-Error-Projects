import gymnasium as gym
import numpy as np

from gymnasium import spaces

class MacroSimulator:
    def __init__(self):
        self.pi_star = 2.0  # target inflation
        self.u_star = 4.0  # natural rate of employment
        self.r_star = 2.0  # central bank targets

        # structural parameters
        self.alpha = 0.5  # demand/unemployment curve
        self.kappa = 0.2  # phillips curve
        # 70% of the current state carries over to the next month, ensuring smooth transitions
        self.rho_u = 0.7  # unemployment momentum
        self.rho_pi = 0.7  # inflation momentum

        self.np_random = np.random

        self.reset()

    def reset(self):
        self.pi = self.pi_star
        self.u = self.u_star
        self.pi_e = self.pi_star
        return self._get_obs()

    def _get_obs(self):
        return {
            "inflation": self.pi,
            "unemployment": self.u
        }

    def step(self, nominal_rate, shock_regime="normal"):
        # compute real rate gap
        real_rate = nominal_rate - self.pi_e
        rate_gap = real_rate - self.r_star

        # define latent shocks
        shock_u, shock_pi = self.np_random.normal(0, 0.1), self.np_random.normal(0, 0.1)

        if shock_regime == "demand":
            # more jobs, prices rise
            shock_u -= 0.5
            shock_pi += 1.0
        elif shock_regime == "supply":
            # stagflation
            shock_u += 0.5
            shock_pi += 1.0

        # demand/unemployment curve
        u_gap = self.u - self.u_star
        new_u = self.u_star + (self.rho_u * u_gap) + (self.alpha * rate_gap) + shock_u
        # unemployment cannot be lower than 0%, so 1% is a compromise
        # 25% was peak of great depression
        self.u = np.clip(new_u, 1.0, 25.0)

        # phillips curve
        pi_gap = self.pi - self.pi_star
        new_u_gap_current = self.u - self.u_star
        new_pi = self.pi_star + (self.rho_pi * pi_gap) - (self.kappa * new_u_gap_current) + shock_pi
        # -5.0% prevents infinite deflation. 30% prevents hyperinflation
        self.pi = np.clip(new_pi, -5.0, 30.0)

        # update expectations
        self.pi_e = (0.5 * self.pi_e) + (0.5 * self.pi)

        return self._get_obs()


class FedEnvBase(gym.Env):
    """
    Wraps the MacroSimulator so the RL algorithm can interact with it
    """
    def __init__(self, llm_dim=4):
        super().__init__()
        self.sim = MacroSimulator()

        # by using a dictionary obs space, the LLM dimension is dynamic
        self.llm_dim = llm_dim

        # action space
        # the federal reserve changes rates in discrete increments
        self.action_mapping = {
            0: -0.75, 1: -0.50, 2: -0.25,  # Rate Cuts
            3:  0.00,                      # Hold Steady
            4:  0.25, 5:  0.50, 6:  0.75   # Rate Hikes
        }
        self.action_space = spaces.Discrete(len(self.action_mapping))

        # observation space
        self.observation_space = spaces.Dict({
            # true macroeconomic indicators: [inflation, unemployment, current policy rate]
            "macro": spaces.Box(
                low=np.array([-10.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([40.0, 30.0, 20.0], dtype=np.float32)
            ),
            # placeholder for llm belief vector
            "llm_belief": spaces.Box(
                low=-1.0, high=1.0, shape=(self.llm_dim,), dtype=np.float32
            )
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.np_random = self.np_random

        # initial policy rate: neutral rate + target inflation = 4%
        self.current_rate = 4.0
        self.t = 0
        self.max_steps = 60

        # randomize shock start and duration to prevent agent memorization
        # this ensures the LSTM learns to detect the shock from signals, not just a fixed clock
        self.shock_start = self.np_random.integers(10, 41)
        self.shock_duration = self.np_random.integers(12, 25)
        self.shock_end = self.shock_start + self.shock_duration

        self.sim.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        macro_obs = np.array([self.sim.pi, self.sim.u, self.current_rate], dtype=np.float32)

        # mock llm vector
        llm_obs = np.zeros(self.llm_dim, dtype=np.float32)

        return {
            "macro": macro_obs,
            "llm_belief": llm_obs
        }

    def step(self, action_idx):
        self.t += 1

        # execute action
        delta_rate = self.action_mapping[action_idx]
        new_rate = self.current_rate + delta_rate

        # policy rate clipping. bound the interest rate between 0 and 20%
        # central banks rarely go negative. 20% is the historical max
        self.current_rate = np.clip(new_rate, 0.0, 20.0)

        # economic engine
        # inject a dynamically scheduled supply shock to test true crisis management
        # by checking against the randomized bounds, we evaluate real regime inference
        regime = "supply" if self.shock_start <= self.t <= self.shock_end else "normal"
        self.sim.step(nominal_rate=self.current_rate, shock_regime=regime)

        # calculate reward
        pi_loss = (self.sim.pi - self.sim.pi_star) ** 2
        u_loss = (self.sim.u - self.sim.u_star) ** 2

        # unemployment penalty
        u_fear_penalty = 5.0 * (self.sim.u - 6.0) ** 2 if self.sim.u > 6.0 else 0.0

        # penalty for erratic actions
        rate_volatitlity_loss = delta_rate ** 2

        # total reward
        reward = -(pi_loss + u_loss + u_fear_penalty + rate_volatitlity_loss)

        # termination
        terminated = False
        truncated = bool(self.t >= self.max_steps)
        info = {
            "regime": regime  # pass hidden state
        }

        return self._get_obs(), float(reward), terminated, truncated, info


class MockLLMObservationWrapper(gym.ObservationWrapper):
    """
    Intercepts the dictionary observation and injects a mock
    'Belief State'.
    """
    def __init__(self, env):
        super().__init__(env)
        self.llm_dim = self.env.unwrapped.llm_dim

    # mocking the LLM Output: [Prob_Normal, Prob_Supply_Shock, Sentiment, Hawkish_Urgency, Uncertainty]
    def observation(self, obs):
        # dynamically align the LLM's mock belief with the true randomized crisis window
        unwrapped_env = self.env.unwrapped
        is_crisis = unwrapped_env.shock_start <= unwrapped_env.t <= unwrapped_env.shock_end

        if is_crisis:
            llm_vector = np.array([0.1, 0.8, -0.9, 0.8, 0.5, 0.0], dtype=np.float32)[:self.llm_dim]
        else:
            llm_vector = np.array([0.8, 0.1, 0.5, -0.2, 0.1, 0.0], dtype=np.float32)[:self.llm_dim]

        noise = unwrapped_env.np_random.normal(0, 0.1, size=self.llm_dim)
        obs["llm_belief"] = np.clip(llm_vector + noise, -1.0, 1.0).astype(np.float32)
        return obs