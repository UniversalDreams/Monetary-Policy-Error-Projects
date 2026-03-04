# ─────────────────────────────────────────────────────────────
# src/config.py  — single source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────

# PPO (MLP)
LR         = 5e-4
N_STEPS    = 240     # 4 episodes × 60 steps
BATCH_SIZE = 60      # 1 episode per mini-batch
N_EPOCHS   = 4
GAMMA      = 0.99

# ── Shared LSTM structural params (same for all conditions) ───
LSTM_N_STEPS    = 1024
LSTM_BATCH_SIZE = 128
LSTM_N_EPOCHS   = 4      # was N_EPOCHS (shared with MLP, keep both)

# ── Baseline: macro-only obs, no LLM signal ───────────────────
# Rationale: simpler obs space → wider clip is safe; higher entropy
# needed because policy must explore rate changes without any hint.
BASELINE_LR             = 3e-4
BASELINE_LR_END         = 5e-5   # ~17% of start (gentler floor than oracle)
BASELINE_LR_DECAY_START = 0.3    # decay begins when 30% training remains (70% warmup)
BASELINE_ENT_COEF       = 0.0
BASELINE_CLIP_RANGE     = 0.10   # wider — obs space is clean, can take bigger steps

# ── Oracle: perfect belief signal ─────────────────────────────
# Rationale: clean gradients → tight clip prevents overfitting;
# low entropy because signal already guides exploration.
ORACLE_LR             = 3e-4
ORACLE_LR_END         = 1e-5    # ~3% of start (aggressive floor)
ORACLE_LR_DECAY_START = 0.4     # decay begins at 40% remaining (current behavior)
ORACLE_ENT_COEF       = 0.0
ORACLE_CLIP_RANGE     = 0.05    # tight — clean gradients, prevent overshoot

# ── LLM offline: noisy DB-lookup belief signal ────────────────
# Rationale: LLM belief has sentiment noise + occasional P_n+P_s violations.
# Lower LR and wider clip absorb gradient noise; higher entropy fights
# premature convergence to policies that ignore the noisy signal.
LLM_LR             = 2e-4
LLM_LR_END         = 1e-5
LLM_LR_DECAY_START = 0.3    # longer stable phase before decay
LLM_ENT_COEF       = 0.0
LLM_CLIP_RANGE     = 0.08   # between baseline and oracle
REWARD_CLIP             = -250.0  # per-step floor — scaled for 120-step episodes (~-30k ceiling)
RATE_VOLATILITY_WEIGHT  = 1.5    # multiplier on delta_rate^2 — discourages erratic moves
SOFT_LANDING_WEIGHT     = 1.0    # peak bonus at (π*, u*) — same units as per-step loss
SOFT_LANDING_SIGMA      = 0.5    # 1σ bandwidth (%) for both inflation and unemployment

# Env
LLM_DIM   = 5
MAX_STEPS = 120       # steps per episode

# Training
DEFAULT_EPISODES      = 500    # live LLM condition
DEFAULT_BASE_EPISODES = 10000   # timestep budget = base_episodes * MAX_STEPS (per env)
N_ENVS            = 4      # parallel envs via DummyVecEnv — matches nb 03
CHECKPOINT_FREQ   = 100_000  # steps between checkpoints
EVAL_SEEDS        = 20     # seeds used for final comparison

# Defaults
DEFAULT_MODEL  = "qwen2.5:14b"
DEFAULT_OUT    = "runs/"
DEFAULT_SEED   = 42
PPO_DEVICE     = "cuda"  # LSTM on GPU; MLP stays on CPU (hardcoded in make_ppo)

# Environment variation
SHOCK_SCALE_MIN  = 0.4   # min supply shock scale multiplier (mild shock)
SHOCK_SCALE_MAX  = 1.6   # max supply shock scale multiplier (severe shock)
P_NO_SHOCK       = 0.30  # fraction of episodes with no supply shock
INIT_STATE_NOISE = 0.5   # std dev of Gaussian noise on initial pi and u

# Offline state-keyed DB
DEFAULT_STATE_DB_PATH  = "data/state_belief_db.json"
CHECKPOINT_EVERY_KEYS  = 10   # save DB every N new LLM calls
