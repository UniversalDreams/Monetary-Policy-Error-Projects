# Monetary Policy Error — CS234

Reinforcement learning for Federal Reserve interest rate policy. An RL agent learns to set rates in a simulated macro economy, augmented by an LLM belief state that signals supply shocks and economic regime.

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
Lightweight macro model (no LLM). Drives inflation and unemployment via:
- **IS curve**: unemployment responds to the real rate gap
- **Phillips curve**: inflation responds to unemployment gap
- **Supply shock**: stagflation regime with simultaneous inflation push and unemployment push; rate cuts are capped during shocks (supply constraints)

Parameters: `alpha=0.5` (IS slope), `kappa=0.2` (Phillips slope), `rho_u=rho_pi=0.7` (momentum).

### `FedEnvBase(gym.Env)`
Wraps `MacroSimulator`. Key properties:
- **Action space**: 7 discrete rate changes — `{±0.75, ±0.50, ±0.25, 0.00}` bps
- **Observation space**: `Dict("macro": Box(3,), "llm_belief": Box(llm_dim,))`
  - `macro`: `[inflation, unemployment, current_rate]`
  - `llm_belief`: zeros by default — filled by a wrapper
- **Episode length**: 120 steps
- **Shock schedule**: randomized per episode — `shock_start ∈ [10, 40]`, `duration ∈ [12, 24]`, `scale ∈ [0.4, 1.6]`; 30% of episodes have no shock

**Reward function:**
```
reward = -(π_loss + u_loss + u_fear_penalty + rate_volatility_loss) + soft_landing_bonus
```
- `π_loss = (π − π*)²`, `u_loss = (u − u*)²`  — squared gaps from targets (π*=2%, u*=4%)
- `u_fear_penalty = 5·(u − 6)²` if `u > 6%` — asymmetric unemployment penalty
- `rate_volatility_loss = 1.5·Δrate²` — penalizes erratic moves
- `soft_landing_bonus = exp(−½·((π_gap/σ)² + (u_gap/σ)²))` — Gaussian peak at targets (σ=0.5%)
- Clipped at `−250` per step

---

## LLM Belief State

The 5-dimensional belief state `[P_normal, P_supply, sentiment, hawkishness, uncertainty]` is populated by one of four wrappers:

| Wrapper | Use case |
|---|---|
| `MockLLMObservationWrapper` | Oracle mock — uses hidden shock state directly |
| `PrecomputedLLMWrapper` | Seed-keyed offline DB — fast RL training |
| `StateKeyedLLMWrapper` | State-keyed offline DB `(π, u, rate)` — generalizes across seeds; nearest-neighbor fallback on miss |
| `LiveLLMWrapper` | Online inference via `DirectLLMAdvisor` — ground truth condition |

### LLM Advisor Classes

**`HierarchicalLLMAdvisor`** — two-layer pipeline, 8 LLM calls per step:

Layer 1 — Economic Actor Panel (sequential, shared context):
1. `commercial_bank` → lending rate, credit standards, lending volume
2. `large_corporations` → price increases, hiring, capex
3. `consumer` → spending change, consumer confidence, savings rate
4. `labor_market` → unemployment, payrolls, wage growth

Each actor sees prior actors' outputs. Commentaries are stitched into a **Monthly Economic Dispatch**.

Layer 2 — Specialist Belief Agents (stateless):
1. `regime_detector` → `P_normal`, `P_supply`
2. `sentiment_analyst` → `sentiment`
3. `hawkdove_analyst` → `hawkishness`
4. `uncertainty_estimator` → `uncertainty` (receives dispatch + all three specialist outputs)

**`DirectLLMAdvisor`** — single LLM call per step. Uses calibration anchor examples in the system prompt for smooth interpolation across the macro state space. Default backend is `AnthropicBackend`.

### LLM Backends

```python
OllamaBackend(model="llama3.2")           # local Ollama, default for hierarchical
AnthropicBackend(model="claude-haiku-4-5-20251001")  # Anthropic API, default for direct
```

---

## Configuration (`src/config.py`)

| Group | Key params |
|---|---|
| PPO (MLP) | `LR=5e-4`, `N_STEPS=240`, `BATCH_SIZE=60` |
| RecurrentPPO (LSTM) | `LSTM_LR=3e-4`, `LSTM_CLIP_RANGE=0.05`, `LSTM_ENT_COEF=0.0` |
| Reward | `REWARD_CLIP=-250`, `RATE_VOLATILITY_WEIGHT=1.5`, `SOFT_LANDING_WEIGHT=1.0`, `SOFT_LANDING_SIGMA=0.5` |
| Environment | `LLM_DIM=5`, `MAX_STEPS=120`, `P_NO_SHOCK=0.30`, `SHOCK_SCALE_MIN/MAX=0.4/1.6` |
| Training | `DEFAULT_EPISODES=500` (live LLM), `DEFAULT_BASE_EPISODES=10000` (base), `N_ENVS=4` |
| Offline DB | `DEFAULT_STATE_DB_PATH="data/state_belief_db.json"`, `CHECKPOINT_EVERY_KEYS=200` |

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Requires Ollama running locally for `OllamaBackend`, or `ANTHROPIC_API_KEY` set for `AnthropicBackend`.
