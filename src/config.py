# ─────────────────────────────────────────────────────────────
# src/config.py  — single source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────

# PPO (MLP)
LR         = 5e-4
N_STEPS    = 240     # 4 episodes × 60 steps
BATCH_SIZE = 60      # 1 episode per mini-batch
N_EPOCHS   = 4
GAMMA      = 0.99

# RecurrentPPO (LSTM) — from nb 03 tuning
LSTM_LR         = 3e-4
LSTM_LR_END     = 1e-5   # floor for linear LR decay (~3% of LSTM_LR)
LSTM_N_STEPS    = 1024
LSTM_BATCH_SIZE = 128
LSTM_ENT_COEF   = 0.0    # no entropy bonus — policy finds good basin without it
LSTM_CLIP_RANGE = 0.05   # tighter clip — prevents catastrophic updates on LSTM
REWARD_CLIP             = -250.0  # per-step floor — scaled for 120-step episodes (~-30k ceiling)
RATE_VOLATILITY_WEIGHT  = 3.0    # multiplier on delta_rate^2 — discourages erratic moves

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
P_NO_SHOCK       = 0.20  # fraction of episodes with no supply shock
INIT_STATE_NOISE = 0.5   # std dev of Gaussian noise on initial pi and u

# Offline state-keyed DB
DEFAULT_STATE_DB_PATH  = "data/state_belief_db.json"
CHECKPOINT_EVERY_KEYS  = 200   # save DB every N new LLM calls
