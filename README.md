# Monetary Policy Error — CS234

Reinforcement learning for Federal Reserve interest rate policy. An RL agent learns to set rates in a simulated macro economy, augmented by an LLM belief state that signals supply shocks and economic regime.

## [View Results](results/README.md)
Training curves, benchmark comparisons, trajectory plots, and COVID-era evaluation across all policy conditions.

---

## Project Structure

```
src/
  config.py        — all hyperparameters (single source of truth)
  fed_env.py       — environment, LLM pipeline, and wrappers
notebooks/
  prototypes/      — development notebooks (01–04)
  data/            — offline belief state databases
runs/              — training checkpoints and logs
```

---

## Environment

### `MacroSimulator`

Lightweight discrete-time macro model. Three coupled equations per step:

### IS Curve

```math
u_{t+1}
=
u^{*}
+
\rho_u (u_t - u^{*})
+
\alpha (r_t - \pi^{e}_t - r^{*})
+
\epsilon^{u}_t
```

### Phillips Curve

```math
\pi_{t+1}
=
\pi^{*}
+
\rho_{\pi}(\pi_t - \pi^{*})
-
\kappa (u_{t+1} - u^{*})
+
\epsilon^{\pi}_t
```

### Inflation Expectations

```math
\pi^{e}_{t+1}
=
0.5\,\pi^{e}_t
+
0.5\,\pi_{t+1}
```

### Supply Shock

Active for duration in [12, 24] steps, scale s in [0.4, 1.6].

```math
\epsilon^{u}_t = \epsilon^{u}_t + 1.0\, s
```

```math
\epsilon^{\pi}_t = \epsilon^{\pi}_t + 2.0\, s
```

```math
g_t = \max(g_t, 0)
```

The rate-gap floor prevents monetary stimulus from reducing unemployment during supply constraints.

---

### Core Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α | 0.5 | IS slope — unemployment sensitivity to real rate gap |
| κ | 0.2 | Phillips slope |
| ρ_u = ρ_π | 0.7 | AR(1) momentum |
| u* | 4 percent | Natural unemployment rate |
| π* | 2 percent | Inflation target |
| r* | 2 percent | Neutral real rate |
| εᵘ, εᵖ | N(0, 0.1) | Base stochastic shocks |

---

## `FedEnvBase(gym.Env)`

Wraps `MacroSimulator`.

### Action space

```
{ ±0.75, ±0.50, ±0.25, 0.00 } percentage points
```

### Observation space

```
Dict(
  "macro": Box(3,),
  "llm_belief": Box(llm_dim,)
)
```

- macro = `[inflation, unemployment, current_rate]`
- llm_belief = zeros by default (filled by wrapper)

Episode length: 120 steps

Shock schedule per episode:

- shock_start in [10, 40]
- duration in [12, 24]
- scale in [0.4, 1.6]
- 30 percent of episodes have no shock

---

## Reward Function

```math
R_t
=
-
\Big[
(\pi_t - \pi^{*})^2
+
(u_t - u^{*})^2
+
P(u_t)
+
1.5(\Delta r_t)^2
\Big]
+
B_t
```

### Unemployment penalty

```math
P(u_t)
=
5(u_t - 6)^2 \quad \text{if } u_t > 6
```

```math
P(u_t) = 0 \quad \text{otherwise}
```

### Soft landing bonus

```math
B_t
=
\exp
\left(
-\frac{1}{2}
\left[
\left(\frac{\pi_t - \pi^{*}}{\sigma}\right)^2
+
\left(\frac{u_t - u^{*}}{\sigma}\right)^2
\right]
\right)
```

```math
\sigma = 0.5
```

Clipped at -250 per step.

---

## Taylor Rule Baseline

Classical heuristic benchmark (`src/benchmark.py`).

```math
r^{*}_t
=
r^{*}
+
\pi_t
+
\phi_{\pi}(\pi_t - \pi^{*})
-
\phi_u (u_t - u^{*})
```

With:

- r* = 2  
- π* = 2  
- u* = 4  
- φ_π = 0.5  
- φ_u = 0.5  

Expanded:

```math
r^{*}_t
=
2
+
\pi_t
+
0.5(\pi_t - 2)
-
0.5(u_t - 4)
```

Desired rate change:

```math
\Delta r^{*}_t = r^{*}_t - r_t
```

Rounded to nearest discrete action in:

```
{ ±0.75, ±0.50, ±0.25, 0.00 }
```

The Taylor Rule fails during supply shocks: inflation signals rate hikes while unemployment is rising — tightening worsens recession dynamics.

---

## LLM Belief State

5-dimensional vector:

```
[P_normal, P_supply, sentiment, hawkishness, uncertainty]
```

### Wrappers

| Wrapper | Use Case |
|----------|----------|
| `MockLLMObservationWrapper` | Oracle mock (uses hidden shock state) |
| `PrecomputedLLMWrapper` | Seed-keyed offline DB |
| `StateKeyedLLMWrapper` | State-keyed DB with nearest-neighbor fallback |
| `LiveLLMWrapper` | Online inference via `DirectLLMAdvisor` |

---

## LLM Advisor Classes

### `HierarchicalLLMAdvisor`

Two-layer pipeline (8 LLM calls per step).

Layer 1 — Economic Actor Panel:

1. `commercial_bank`
2. `large_corporations`
3. `consumer`
4. `labor_market`

Outputs combined into a Monthly Economic Dispatch.

Layer 2 — Specialist Agents:

1. `regime_detector`
2. `sentiment_analyst`
3. `hawkdove_analyst`
4. `uncertainty_estimator`

---

### `DirectLLMAdvisor`

Single LLM call per step with calibration anchors in the system prompt.  
Default backend: `AnthropicBackend`.

---

## LLM Backends

```python
OllamaBackend(model="llama3.2")
AnthropicBackend(model="claude-haiku-4-5-20251001")
```

---

## Configuration (`src/config.py`)

### PPO (MLP)

- LR = 5e-4  
- N_STEPS = 240  
- BATCH_SIZE = 60  

### RecurrentPPO (Shared)

- LSTM_N_STEPS = 1024  
- LSTM_BATCH_SIZE = 128  
- LSTM_N_EPOCHS = 4  

### Reward

- REWARD_CLIP = -250  
- RATE_VOLATILITY_WEIGHT = 1.5  
- SOFT_LANDING_WEIGHT = 1.0  
- SOFT_LANDING_SIGMA = 0.5  

### Environment

- LLM_DIM = 5  
- MAX_STEPS = 120  
- P_NO_SHOCK = 0.30  
- SHOCK_SCALE_MIN = 0.4  
- SHOCK_SCALE_MAX = 1.6  

### Training

- DEFAULT_EPISODES = 500  
- DEFAULT_BASE_EPISODES = 10000  
- N_ENVS = 4  

### Offline DB

- DEFAULT_STATE_DB_PATH = "data/state_belief_db.json"  
- CHECKPOINT_EVERY_KEYS = 10  

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Requires:

- Ollama running locally for `OllamaBackend`
- `ANTHROPIC_API_KEY` set for `AnthropicBackend`