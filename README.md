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
Lightweight discrete-time macro model. Three coupled equations per step:

**IS curve** — unemployment responds to the real rate gap:

$$u_{t+1} = u^{*} + \rho_u(u_t - u^{*}) + \alpha(r_t - \pi^{e}_t - r^{*}) + \epsilon^{u}_t$$

**Phillips curve** — inflation responds to the updated unemployment gap:

$$\pi_{t+1} = \pi^{*} + \rho_{\pi}(\pi_t - \pi^{*}) - \kappa(u_{t+1} - u^{*}) + \epsilon^{\pi}_t$$

**Inflation expectations** — adaptive (50/50 lag):

$$\pi^{e}_{t+1} = 0.5\,\pi^{e}_t + 0.5\,\pi_{t+1}$$

**Supply shock** (active for `duration ∈ [12, 24]` steps, scale $s \in [0.4, 1.6]$):

$$\epsilon^{u}_t \mathrel{+}= 1.0\cdot s, \quad \epsilon^{\pi}_t \mathrel{+}= 2.0\cdot s, \quad \text{real rate gap} = \max(\text{real rate gap},\; 0)$$

The rate-gap floor prevents monetary stimulus from reducing unemployment during supply constraints.

| Parameter | Value | Meaning |
|---|---|---|
| $\alpha$ | 0.5 | IS slope — sensitivity of unemployment to real rate gap |
| $\kappa$ | 0.2 | Phillips slope |
| $\rho_u = \rho_{\pi}$ | 0.7 | AR(1) momentum for unemployment and inflation |
| $u^{*}$ | 4% | Unemployment target (natural rate) |
| $\pi^{*}$ | 2% | Inflation target |
| $r^{*}$ | 2% | Neutral real rate |
| $\epsilon^{u}, \epsilon^{\pi}$ | $\mathcal{N}(0,\, 0.1)$ | Base stochastic shocks |

### `FedEnvBase(gym.Env)`
Wraps `MacroSimulator`. Key properties:
- **Action space**: 7 discrete rate changes — `{±0.75, ±0.50, ±0.25, 0.00}` pp (percentage points)
- **Observation space**: `Dict("macro": Box(3,), "llm_belief": Box(llm_dim,))`
  - `macro`: `[inflation, unemployment, current_rate]`
  - `llm_belief`: zeros by default — filled by a wrapper
- **Episode length**: 120 steps
- **Shock schedule**: randomized per episode — `shock_start ∈ [10, 40]`, `duration ∈ [12, 24]`, `scale ∈ [0.4, 1.6]`; 30% of episodes have no shock

**Reward function:**

$$R_t = -\bigl[(\pi_t - \pi^{*})^{2} + (u_t - u^{*})^{2} + P(u_t) + 1.5\,\Delta r_t^{2}\bigr] + B_t$$

$$P(u_t) = \begin{cases} 5\,(u_t - 6)^{2} & \text{if } u_t > 6\% \\ 0 & \text{otherwise} \end{cases}$$

$$B_t = \exp\!\left(-\tfrac{1}{2}\left[\left(\tfrac{\pi_t - \pi^{*}}{\sigma}\right)^{\!2} + \left(\tfrac{u_t - u^{*}}{\sigma}\right)^{\!2}\right]\right), \quad \sigma = 0.5\%$$

| Symbol | Meaning |
|---|---|
| $\pi_t$ | inflation rate at step $t$ |
| $u_t$ | unemployment rate at step $t$ |
| $\Delta r_t$ | rate change chosen at step $t$ |
| $\sigma = 0.5\%$ | soft-landing bandwidth |
| $\pi^{*}, u^{*}, r^{*}$ | targets — see MacroSimulator parameter table above |

Clipped at $-250$ per step.

### Taylor Rule Baseline

The classical heuristic policy used as a performance benchmark (`src/benchmark.py`).

$$r^{*}_t = r^{*} + \pi_t + \phi_{\pi}(\pi_t - \pi^{*}) - \phi_u(u_t - u^{*})$$

With $r^{*} = 2\%$, $\pi^{*} = 2\%$, $u^{*} = 4\%$, $\phi_{\pi} = 0.5$, $\phi_u = 0.5$, this expands to:

$$r^{*}_t = 2 + \pi_t + 0.5(\pi_t - 2) - 0.5(u_t - 4)$$

The desired rate change $\Delta r^{*}_t = r^{*}_t - r_t$ is then rounded to the nearest discrete action in $\{{\pm0.75, \pm0.50, \pm0.25, 0.00}\}$.

The Taylor Rule fails during supply shocks: high inflation signals a rate hike, but the shock is simultaneously pushing unemployment up — so hiking worsens the recession.

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
| RecurrentPPO shared | `LSTM_N_STEPS=1024`, `LSTM_BATCH_SIZE=128`, `LSTM_N_EPOCHS=4` |
| RecurrentPPO per-condition | see table below |
| Reward | `REWARD_CLIP=-250`, `RATE_VOLATILITY_WEIGHT=1.5`, `SOFT_LANDING_WEIGHT=1.0`, `SOFT_LANDING_SIGMA=0.5` |
| Environment | `LLM_DIM=5`, `MAX_STEPS=120`, `P_NO_SHOCK=0.30`, `SHOCK_SCALE_MIN/MAX=0.4/1.6` |
| Training | `DEFAULT_EPISODES=500` (live LLM), `DEFAULT_BASE_EPISODES=10000` (base), `N_ENVS=4` |
| Offline DB | `DEFAULT_STATE_DB_PATH="data/state_belief_db.json"`, `CHECKPOINT_EVERY_KEYS=10` |

**Per-condition LSTM hyperparameters** (LSTM policy only):

| Condition | LR | LR_END | LR_DECAY_START | ENT_COEF | CLIP_RANGE |
|---|---|---|---|---|---|
| baseline | 3e-4 | 5e-5 | 0.3 | 0.0 | 0.10 |
| oracle   | 3e-4 | 1e-5 | 0.4 | 0.0 | 0.05 |
| llm      | 2e-4 | 1e-5 | 0.3 | 0.0 | 0.08 |

`LR_DECAY_START` is the fraction of training remaining when LR decay begins
(e.g. 0.3 → decay starts at 70% through training).

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Requires Ollama running locally for `OllamaBackend`, or `ANTHROPIC_API_KEY` set for `AnthropicBackend`.
