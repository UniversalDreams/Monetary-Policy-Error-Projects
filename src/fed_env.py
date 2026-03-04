import json
import os
import sys
import gymnasium as gym
import numpy as np

from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
import config

_REWARD_CLIP            = config.REWARD_CLIP
_RATE_VOLATILITY_WEIGHT = config.RATE_VOLATILITY_WEIGHT
_SOFT_LANDING_WEIGHT    = config.SOFT_LANDING_WEIGHT
_SOFT_LANDING_SIGMA     = config.SOFT_LANDING_SIGMA
_INIT_STATE_NOISE       = config.INIT_STATE_NOISE


# ─────────────────────────────────────────────────────────────
# LLM BACKENDS
# ─────────────────────────────────────────────────────────────

class LLMBackend:
    """Abstract LLM call interface — swap implementations without changing agent code."""
    def __call__(self, system_prompt: str, messages: list) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Local Ollama via /api/chat. Default backend."""
    def __init__(self, model="llama3.2", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def __call__(self, system_prompt: str, messages: list) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": False
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API."""
    def __init__(self, model="claude-haiku-4-5-20251001"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model

    def __call__(self, system_prompt: str, messages: list) -> str:
        response = self.client.messages.create(
            model=self.model, max_tokens=512,
            system=system_prompt, messages=messages
        )
        return response.content[0].text


# ─────────────────────────────────────────────────────────────
# JSON PARSING UTILITY
# ─────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON; returns error dict on failure."""
    text = raw.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                inner.append(line)
        text = "\n".join(inner)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "parse_failed"}


# ─────────────────────────────────────────────────────────────
# ECONOMIC ACTOR CONFIGS
# ─────────────────────────────────────────────────────────────

ECONOMIC_ACTORS = {
    "commercial_bank": {
        "name": "Commercial Bank",
        "system_prompt": (
            "You represent the US commercial banking sector. "
            "You will receive the current macro briefing with EXACT numbers for inflation, unemployment, and the Fed rate. "
            "These numbers are ground truth — base your response strictly on them, do not invent stress or crises not supported by the data.\n\n"
            "CALIBRATION GUIDE:\n"
            "- Inflation ~2%, unemployment ~4%, rate ~4%: stable economy, normal credit standards, healthy lending\n"
            "- Inflation >5%, unemployment rising: tighten credit standards, reduce lending volume\n"
            "- Unemployment >6%, falling inflation: loosen standards, support growth\n\n"
            "Output ONLY this JSON (no markdown, no extra text):\n"
            '{"lending_rate": <float>, "credit_standards": "<tight|normal|loose>", '
            '"new_lending_volume_index": <float 0-100>, "commentary": "<1-2 sentences describing current conditions>"}'
        ),
    },
    "large_corporations": {
        "name": "Large Corporations",
        "system_prompt": (
            "You represent S&P 500 corporations. "
            "You will receive the current macro briefing with EXACT numbers. "
            "These numbers are ground truth — base your response strictly on them, do not invent stress or crises not supported by the data.\n\n"
            "CALIBRATION GUIDE:\n"
            "- Inflation ~2%, unemployment ~4%: modest price increases (~2%), steady hiring, normal capex\n"
            "- Inflation >5%: larger price increases, cautious hiring\n"
            "- Unemployment >6%: freeze hiring, cut capex\n\n"
            "Output ONLY this JSON (no markdown, no extra text):\n"
            '{"price_increases_pct": <float>, "hiring_thousands": <float>, '
            '"capex_change_pct": <float>, "commentary": "<1-2 sentences describing current conditions>"}'
        ),
    },
    "consumer": {
        "name": "Composite Consumer",
        "system_prompt": (
            "You represent aggregate US household spending. "
            "You will receive the current macro briefing with EXACT numbers. "
            "These numbers are ground truth — base your response strictly on them, do not invent stress or crises not supported by the data.\n\n"
            "CALIBRATION GUIDE:\n"
            "- Inflation ~2%, unemployment ~4%: confidence ~65, spending growth ~2%, savings ~5%\n"
            "- Inflation >5%: confidence drops, spending slows, savings rise\n"
            "- Unemployment >6%: confidence drops sharply, spending contracts\n\n"
            "Output ONLY this JSON (no markdown, no extra text):\n"
            '{"spending_change_pct": <float>, "consumer_confidence": <float 0-100>, '
            '"savings_rate_pct": <float>, "commentary": "<1-2 sentences describing current conditions>"}'
        ),
    },
    "labor_market": {
        "name": "Labor Market",
        "system_prompt": (
            "You synthesize sector hiring decisions into aggregate labor market statistics. "
            "You will receive the current macro briefing with EXACT numbers. "
            "These numbers are ground truth — your output must be consistent with the reported unemployment rate.\n\n"
            "CALIBRATION GUIDE:\n"
            "- Unemployment ~4%: payrolls +150k to +250k/month, wage growth ~3%\n"
            "- Unemployment rising toward 6%: payrolls flat or negative, wage growth slowing\n"
            "- Unemployment >7%: payrolls deeply negative, wage growth <2%\n\n"
            "Output ONLY this JSON (no markdown, no extra text):\n"
            '{"unemployment_rate": <float>, "nonfarm_payrolls_change_thousands": <float>, '
            '"wage_growth_pct": <float>, "commentary": "<1-2 sentences describing current conditions>"}'
        ),
    },
}


# ─────────────────────────────────────────────────────────────
# SPECIALIST AGENT CONFIGS
# ─────────────────────────────────────────────────────────────

SPECIALIST_AGENTS = {
    "regime_detector": {
        "system_prompt": (
            "You are a macroeconomic regime analyst at the Federal Reserve.\n"
            "Read the Monthly Economic Dispatch and assess the probability of a supply shock / stagflation regime.\n\n"
            "CALIBRATION — use these thresholds strictly:\n"
            "  NORMAL regime (P_supply < 0.2):\n"
            "    - Inflation near 2%, unemployment near 4%\n"
            "    - Banks lending normally, corporations hiring steadily, consumers confident\n"
            "  AMBIGUOUS (P_supply 0.2-0.5):\n"
            "    - Inflation rising above 3% but unemployment still low\n"
            "    - Could be demand-driven, not yet stagflation\n"
            "  SUPPLY SHOCK (P_supply > 0.5):\n"
            "    - BOTH inflation above 4% AND unemployment above 5% simultaneously\n"
            "    - Banks tightening credit, corporations raising prices while also laying off workers\n"
            "    - Consumer confidence falling while prices rise\n\n"
            "IMPORTANT: P_normal + P_supply must sum to 1.0.\n"
            "Do NOT assign P_supply > 0.3 unless there is clear evidence of SIMULTANEOUS high inflation AND rising unemployment.\n\n"
            'Output ONLY this JSON:\n{"P_normal": <float 0-1>, "P_supply": <float 0-1>, "reasoning": "<1 sentence>"}'
        ),
    },
    "sentiment_analyst": {
        "system_prompt": (
            "You are a market sentiment analyst. Read the Monthly Economic Dispatch\n"
            "and rate overall economic sentiment from -1 (crisis/recession) to +1 (boom).\n\n"
            "Consider: consumer confidence, corporate hiring, credit availability, spending trends.\n\n"
            "CALIBRATION — anchor your scores:\n"
            "  DEFLATION (inflation < 0%): negative sentiment (-0.3 to -0.8).\n"
            "    Falling prices signal weak demand and recession risk — not a positive.\n"
            "  BELOW TARGET (0-1.5%): mildly negative to neutral (-0.1 to -0.4).\n"
            "  NORMAL (~2%, ~4% unemployment): moderately positive (+0.3 to +0.7).\n"
            "  OVERHEATING (inflation > 4%, low unemployment): mildly negative (-0.1 to -0.4).\n"
            "    Strong activity, but inflation erodes real incomes and rate hikes loom.\n"
            "  STAGFLATION (high inflation + high unemployment): strongly negative (-0.5 to -1.0).\n\n"
            'Output ONLY this JSON:\n{"sentiment": <float -1 to 1>, "reasoning": "<1 sentence>"}'
        ),
    },
    "hawkdove_analyst": {
        "system_prompt": (
            "You are a monetary policy specialist advising the FOMC.\n"
            "Read the Monthly Economic Dispatch and assess urgency for rate action.\n"
            "+1 = strongly hawkish (inflation crisis, raise rates now)\n"
            "-1 = strongly dovish (recession/deflation risk, cut rates aggressively)\n"
            " 0 = hold steady\n\n"
            "CALIBRATION — anchor your scores:\n"
            "  DEFLATION (inflation < 0%): strongly dovish (-0.6 to -1.0).\n"
            "    Deflation with elevated rates means extremely tight real rates — cut aggressively.\n"
            "  BELOW TARGET (0-1.5%): moderately dovish (-0.3 to -0.6).\n"
            "  ON TARGET (~2%): neutral, adjust slightly based on unemployment.\n"
            "  OVERHEATING (inflation 3-5%, low unemployment): moderately hawkish (+0.3 to +0.6).\n"
            "  INFLATION CRISIS (inflation > 5%): strongly hawkish (+0.6 to +1.0).\n"
            "  STAGFLATION dilemma: slightly hawkish (+0.2 to +0.4) but NOT aggressively —\n"
            "    raising rates in a supply shock worsens unemployment.\n\n"
            'Output ONLY this JSON:\n{"hawkishness": <float -1 to 1>, "reasoning": "<1 sentence>"}'
        ),
    },
    "uncertainty_estimator": {
        "system_prompt": (
            "You are a risk assessment officer. Given the Economic Dispatch AND the three\n"
            "specialist analyses, assess total epistemic uncertainty about the current regime.\n\n"
            "0 = certain (all analysts agree, data crystal clear)\n"
            "1 = maximum uncertainty (analysts contradict, unusual regime)\n\n"
            "High uncertainty expected in:\n"
            "  - Early stages of supply shocks (looks like demand inflation initially)\n"
            "  - Deflation onset (hard to distinguish from low-inflation normal)\n"
            "  - Mixed signals: low/negative inflation but rising unemployment\n\n"
            'Output ONLY this JSON:\n{"uncertainty": <float 0 to 1>, "reasoning": "<1 sentence>"}'
        ),
    },
}


# ─────────────────────────────────────────────────────────────
# ECONOMIC ACTOR
# ─────────────────────────────────────────────────────────────

class EconomicActor:
    """
    Wraps one economic sector agent. Adapted from initial_env.py::LLMAgent.
    Maintains rolling conversation history (last 6 messages = 3 rounds)
    within an episode, reset between episodes.
    """
    def __init__(self, actor_id: str, config: dict, backend: LLMBackend):
        self.actor_id = actor_id
        self.name = config["name"]
        self.system_prompt = config["system_prompt"]
        self.backend = backend
        self.history: list = []

    def decide(self, macro_summary: str, other_actor_outputs: dict) -> dict:
        """Call LLM with macro context + other actors' decisions."""
        user_message = self._build_context(macro_summary, other_actor_outputs)
        messages = [*self.history[-6:], {"role": "user", "content": user_message}]
        raw = self.backend(self.system_prompt, messages)
        decision = _parse_json(raw)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": raw})
        return decision

    def _build_context(self, macro_summary: str, others: dict) -> str:
        ctx = f"MONTHLY MACRO BRIEFING:\n{macro_summary}\n"
        if others:
            ctx += "\nOTHER SECTOR DECISIONS THIS MONTH:\n"
            for aid, dec in others.items():
                ctx += f"\n[{aid}]: {json.dumps(dec)[:400]}\n"
        ctx += "\nOutput only your JSON decision."
        return ctx

    def reset_history(self):
        self.history = []


# ─────────────────────────────────────────────────────────────
# SPECIALIST AGENT
# ─────────────────────────────────────────────────────────────

class SpecialistAgent:
    """Stateless per-timestep specialist analyzer."""
    def __init__(self, agent_id: str, config: dict, backend: LLMBackend):
        self.agent_id = agent_id
        self.system_prompt = config["system_prompt"]
        self.backend = backend

    def analyze(self, user_message: str) -> dict:
        raw = self.backend(self.system_prompt, [{"role": "user", "content": user_message}])
        return _parse_json(raw)


# ─────────────────────────────────────────────────────────────
# HIERARCHICAL LLM ADVISOR
# ─────────────────────────────────────────────────────────────

class HierarchicalLLMAdvisor:
    """Orchestrates both layers. Public interface for notebook and PrecomputedLLMWrapper."""
    def __init__(self, backend: LLMBackend = None):
        if backend is None:
            backend = OllamaBackend()
        self.actors = {aid: EconomicActor(aid, cfg, backend)
                       for aid, cfg in ECONOMIC_ACTORS.items()}
        self.specialists = {sid: SpecialistAgent(sid, cfg, backend)
                            for sid, cfg in SPECIALIST_AGENTS.items()}

    def get_belief_state(self, pi: float, u: float, rate: float, regime: str) -> tuple:
        """Full pipeline for one timestep. Returns (dispatch_text, belief_state)."""
        macro_summary = f"Inflation: {pi:.1f}%, Unemployment: {u:.1f}%, Fed Rate: {rate:.2f}%"

        # Layer 1: Economic actors in dependency order
        actor_order = ["commercial_bank", "large_corporations", "consumer", "labor_market"]
        actor_outputs = {}
        for aid in actor_order:
            actor_outputs[aid] = self.actors[aid].decide(macro_summary, actor_outputs)

        # Stitch commentaries into economic dispatch
        dispatch = f"Monthly Economic Dispatch — {macro_summary}\n"
        for aid in actor_order:
            name = ECONOMIC_ACTORS[aid]["name"]
            commentary = actor_outputs[aid].get("commentary", "No comment.")
            dispatch += f"\n[{name}]: {commentary}"

        # Layer 2: Specialists analyze dispatch (first 3 in parallel conceptually, run sequentially)
        regime_out    = self.specialists["regime_detector"].analyze(dispatch)
        sentiment_out = self.specialists["sentiment_analyst"].analyze(dispatch)
        hawkdove_out  = self.specialists["hawkdove_analyst"].analyze(dispatch)

        uncertainty_input = json.dumps({
            "dispatch": dispatch[:1000],
            "regime_analysis": regime_out,
            "sentiment": sentiment_out,
            "hawkishness": hawkdove_out
        })
        uncertainty_out = self.specialists["uncertainty_estimator"].analyze(uncertainty_input)

        # Assemble belief state
        belief = [
            float(regime_out.get("P_normal", 0.5)),
            float(regime_out.get("P_supply", 0.5)),
            float(sentiment_out.get("sentiment", 0.0)),
            float(hawkdove_out.get("hawkishness", 0.0)),
            float(uncertainty_out.get("uncertainty", 0.5)),
        ]
        belief = [max(-1.0, min(1.0, v)) for v in belief]
        return dispatch, belief

    def reset_episode(self):
        """Reset actor conversation histories between episodes."""
        for actor in self.actors.values():
            actor.reset_history()
        # Specialists are stateless — no reset needed


# ─────────────────────────────────────────────────────────────
# DIRECT LLM ADVISOR
# ─────────────────────────────────────────────────────────────

DIRECT_SYSTEM_PROMPT = """You are a Federal Reserve macroeconomic analyst. Given current economic
indicators, output a 5-dimensional belief state vector quantifying the economic regime.

BELIEF STATE DIMENSIONS:
- P_normal: probability of healthy/normal economy [0, 1]
- P_supply: probability of active supply shock / stagflation [0, 1]  (P_normal + P_supply = 1.0)
- sentiment: overall economic sentiment [-1 = crisis, +1 = boom]
- hawkishness: urgency for rate hikes [-1 = cut aggressively, +1 = hike aggressively]
- uncertainty: epistemic uncertainty about the regime [0 = certain, 1 = maximum uncertainty]

CALIBRATION EXAMPLES (interpolate smoothly between these):
pi=-1.0%, u=4.5%: {"P_normal":0.60,"P_supply":0.05,"sentiment":-0.6,"hawkishness":-0.8,"uncertainty":0.25}
pi=0.5%,  u=4.0%: {"P_normal":0.80,"P_supply":0.05,"sentiment":-0.2,"hawkishness":-0.5,"uncertainty":0.15}
pi=2.0%,  u=4.0%: {"P_normal":0.92,"P_supply":0.08,"sentiment":0.5,"hawkishness":-0.1,"uncertainty":0.10}
pi=3.0%,  u=4.0%: {"P_normal":0.78,"P_supply":0.22,"sentiment":0.3,"hawkishness":0.3,"uncertainty":0.20}
pi=4.0%,  u=4.5%: {"P_normal":0.55,"P_supply":0.45,"sentiment":0.0,"hawkishness":0.3,"uncertainty":0.40}
pi=5.0%,  u=5.0%: {"P_normal":0.28,"P_supply":0.72,"sentiment":-0.4,"hawkishness":0.2,"uncertainty":0.55}
pi=6.0%,  u=5.5%: {"P_normal":0.12,"P_supply":0.88,"sentiment":-0.6,"hawkishness":0.2,"uncertainty":0.45}
pi=7.0%,  u=6.0%: {"P_normal":0.05,"P_supply":0.95,"sentiment":-0.8,"hawkishness":0.1,"uncertainty":0.35}

KEY RULES:
- P_normal + P_supply must equal exactly 1.0
- Stagflation = high inflation AND high unemployment simultaneously
- Rising inflation alone (low unemployment) = demand shock, NOT supply shock
- Hawkishness near 0 during stagflation — raising rates aggressively worsens unemployment
- Uncertainty peaks early in a supply shock (hard to distinguish from demand inflation)
- DEFLATION (pi < 0%): sentiment must be negative; hawkishness must be strongly negative (-0.6 to -1.0).
  Deflation is not a positive signal — it indicates weak demand and recession risk.
  Real rates are already elevated when pi < 0; cutting is the priority.
- High fed rate with low/negative inflation = very tight policy = reinforce dovish signal.

Output ONLY valid JSON, no markdown, no explanation:
{"P_normal":x,"P_supply":x,"sentiment":x,"hawkishness":x,"uncertainty":x}"""


class DirectLLMAdvisor:
    """
    Single-call LLM advisor. Replaces the 8-call hierarchical pipeline.
    Reads macro numbers directly and outputs the 5-dim belief state
    using calibration anchor examples for smooth interpolation.
    Same public interface as HierarchicalLLMAdvisor.
    """
    def __init__(self, backend: LLMBackend = None):
        if backend is None:
            backend = AnthropicBackend()
        self.backend = backend

    def get_belief_state(self, pi: float, u: float, rate: float, regime: str) -> tuple:
        """Single LLM call. Returns (context_str, belief_state)."""
        user_message = (
            f"Current indicators: Inflation={pi:.2f}%, Unemployment={u:.2f}%, Fed Rate={rate:.2f}%\n"
            f"Output the belief state JSON."
        )
        raw = self.backend(DIRECT_SYSTEM_PROMPT, [{"role": "user", "content": user_message}])
        result = _parse_json(raw)

        belief = [
            float(result.get("P_normal", 0.5)),
            float(result.get("P_supply", 0.5)),
            float(result.get("sentiment", 0.0)),
            float(result.get("hawkishness", 0.0)),
            float(result.get("uncertainty", 0.5)),
        ]
        belief = [max(-1.0, min(1.0, v)) for v in belief]
        context = f"Direct: pi={pi:.2f}% u={u:.2f}% rate={rate:.2f}%"
        return context, belief

    def reset_episode(self):
        pass  # stateless — no history to reset


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
        noise_pi = self.np_random.normal(0, _INIT_STATE_NOISE)
        noise_u  = self.np_random.normal(0, _INIT_STATE_NOISE)
        self.pi   = np.clip(self.pi_star + noise_pi, 0.0, 10.0)
        self.u    = np.clip(self.u_star  + noise_u,  1.0, 10.0)
        self.pi_e = self.pi
        return self._get_obs()

    def _get_obs(self):
        return {
            "inflation": self.pi,
            "unemployment": self.u
        }

    def step(self, nominal_rate, shock_regime="normal", shock_scale=1.0):
        # compute real rate gap
        real_rate = nominal_rate - self.pi_e
        rate_gap = real_rate - self.r_star

        # define latent shocks
        shock_u, shock_pi = self.np_random.normal(0, 0.1), self.np_random.normal(0, 0.1)

        if shock_regime == "demand":
            # more jobs, prices rise
            shock_u -= 0.5 * shock_scale
            shock_pi += 1.0 * shock_scale
        elif shock_regime == "supply":
            # stagflation: moderate unemployment push + strong inflation push
            shock_u += 1.0 * shock_scale
            shock_pi += 2.0 * shock_scale
            # supply constraints prevent hiring even with loose monetary policy —
            # cap the stimulative effect of a negative real rate during the shock
            rate_gap = max(rate_gap, 0.0)

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
        self.max_steps = 120

        # randomize shock start and duration to prevent agent memorization
        # this ensures the LSTM learns to detect the shock from signals, not just a fixed clock
        if self.np_random.random() < config.P_NO_SHOCK:
            # no-shock episode: set shock_start beyond episode length so it never triggers
            self.shock_start    = self.max_steps + 1
            self.shock_duration = 0
            self.shock_scale    = 0.0
        else:
            self.shock_start    = self.np_random.integers(10, 41)
            self.shock_duration = self.np_random.integers(12, 25)
            self.shock_scale    = float(self.np_random.uniform(
                                      config.SHOCK_SCALE_MIN, config.SHOCK_SCALE_MAX))
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
        self.sim.step(nominal_rate=self.current_rate,
                      shock_regime=regime,
                      shock_scale=self.shock_scale)

        # calculate reward
        pi_loss = (self.sim.pi - self.sim.pi_star) ** 2
        u_loss = (self.sim.u - self.sim.u_star) ** 2

        # unemployment penalty
        u_fear_penalty = 5.0 * (self.sim.u - 6.0) ** 2 if self.sim.u > 6.0 else 0.0

        # penalty for erratic actions
        rate_volatitlity_loss = _RATE_VOLATILITY_WEIGHT * delta_rate ** 2

        # soft landing bonus: Gaussian peak at (π*, u*) rewards precision near targets
        pi_gap = self.sim.pi - self.sim.pi_star
        u_gap  = self.sim.u  - self.sim.u_star
        soft_landing_bonus = _SOFT_LANDING_WEIGHT * float(np.exp(
            -0.5 * ((pi_gap / _SOFT_LANDING_SIGMA) ** 2 + (u_gap / _SOFT_LANDING_SIGMA) ** 2)
        ))

        # total reward (clipped to prevent catastrophic episodes from drowning gradients)
        reward = max(-(pi_loss + u_loss + u_fear_penalty + rate_volatitlity_loss)
                     + soft_landing_bonus, _REWARD_CLIP)

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
        scale = getattr(unwrapped_env, 'shock_scale', 1.0)

        if is_crisis:
            p_supply    = float(np.clip(0.5 + 0.3 * scale, 0.5, 0.95))
            hawkishness = float(np.clip(0.2 + 0.5 * scale, 0.2, 0.9))
            llm_vector  = np.array([1 - p_supply, p_supply, -0.6 * scale,
                                     hawkishness, 0.3 + 0.1 * scale], dtype=np.float32)
        else:
            llm_vector  = np.array([0.8, 0.1, 0.5, -0.2, 0.1], dtype=np.float32)

        llm_vector = llm_vector[:self.llm_dim]
        noise = unwrapped_env.np_random.normal(0, 0.05, size=self.llm_dim)
        obs["llm_belief"] = np.clip(llm_vector + noise, -1.0, 1.0).astype(np.float32)
        return obs


class PrecomputedLLMWrapper(gym.ObservationWrapper):
    """
    Replaces MockLLMObservationWrapper for RL training.
    Reads belief states from a pre-built offline database.
    """
    def __init__(self, env, db_path: str):
        super().__init__(env)
        with open(db_path) as f:
            self.db = json.load(f)
        self._current_seed = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_seed = str(seed)
        return self.observation(obs), info

    def observation(self, obs):
        try:
            belief = self.db["episodes"][self._current_seed]["steps"][
                str(self.env.unwrapped.t)]["belief_state"]
            obs["llm_belief"] = np.array(belief, dtype=np.float32)
        except (KeyError, TypeError):
            obs["llm_belief"] = np.zeros(self.env.unwrapped.llm_dim, dtype=np.float32)
        return obs


class StateKeyedLLMWrapper(gym.ObservationWrapper):
    """
    State-keyed offline belief DB wrapper.
    Looks up belief state by (pi, u, rate) rounded to 0.1% resolution,
    enabling any episode trajectory to find a match regardless of seed.
    Falls back to zeros on cache miss and tracks the miss count.
    """
    def __init__(self, env, db_path: str):
        super().__init__(env)
        with open(db_path) as f:
            db = json.load(f)
        self._states = db["states"]   # {"pi_u_rate": [5 floats]}
        self._misses = 0

        # Build NN lookup arrays for nearest-neighbor fallback on cache miss
        if self._states:
            coords, beliefs = [], []
            for key, belief in self._states.items():
                pi_s, u_s, rate_s = key.split("_")
                coords.append([float(pi_s), float(u_s), float(rate_s)])
                beliefs.append(belief)
            self._nn_coords  = np.array(coords,  dtype=np.float32)  # (N, 3)
            self._nn_beliefs = np.array(beliefs, dtype=np.float32)  # (N, 5)
        else:
            self._nn_coords  = None
            self._nn_beliefs = None

    def _make_key(self, pi: float, u: float, rate: float) -> str:
        return f"{pi:.1f}_{u:.1f}_{rate:.2f}"

    def observation(self, obs):
        pi, u, rate = obs["macro"]
        key = self._make_key(float(pi), float(u), float(rate))
        belief = self._states.get(key)
        if belief is None:
            self._misses += 1
            if self._nn_coords is not None:
                query = np.array([float(pi), float(u), float(rate)], dtype=np.float32)
                dists = np.sum((self._nn_coords - query) ** 2, axis=1)
                obs["llm_belief"] = self._nn_beliefs[np.argmin(dists)].copy()
            else:
                obs["llm_belief"] = np.zeros(self.env.unwrapped.llm_dim, dtype=np.float32)
        else:
            obs["llm_belief"] = np.array(belief, dtype=np.float32)
        return obs


class LiveLLMWrapper(gym.ObservationWrapper):
    """
    Live LLM inference wrapper for online RL training.
    Calls DirectLLMAdvisor at each step so the belief state always reflects
    the actual macro state, not a precomputed counterfactual trajectory.
    Replaces PrecomputedLLMWrapper for the paper's training experiments.
    """
    def __init__(self, env, advisor: DirectLLMAdvisor = None):
        super().__init__(env)
        if advisor is None:
            advisor = DirectLLMAdvisor()
        self.advisor = advisor

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.advisor.reset_episode()
        return self.observation(obs), info

    def observation(self, obs):
        pi, u, rate = obs["macro"]
        _, belief = self.advisor.get_belief_state(float(pi), float(u), float(rate), "unknown")
        obs["llm_belief"] = np.array(belief, dtype=np.float32)
        return obs