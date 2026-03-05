# src/config.py  — single source of truth for all hyperparameters

# PPO (MLP)
LR         = 5e-4
N_STEPS    = 240
BATCH_SIZE = 60
N_EPOCHS   = 4
GAMMA      = 0.99

# LSTM (shared structural params)
LSTM_N_STEPS    = 1024
LSTM_BATCH_SIZE = 128
LSTM_N_EPOCHS   = 4

# Baseline condition
BASELINE_LR             = 3e-4
BASELINE_LR_END         = 5e-5
BASELINE_LR_DECAY_START = 0.3
BASELINE_ENT_COEF       = 0.0
BASELINE_CLIP_RANGE     = 0.10

# Oracle condition
ORACLE_LR             = 3e-4
ORACLE_LR_END         = 1e-5
ORACLE_LR_DECAY_START = 0.4
ORACLE_ENT_COEF       = 0.0
ORACLE_CLIP_RANGE     = 0.05

# LLM offline condition
LLM_LR             = 2e-4
LLM_LR_END         = 1e-5
LLM_LR_DECAY_START = 0.3
LLM_ENT_COEF       = 0.0
LLM_CLIP_RANGE     = 0.08

# Reward shaping
REWARD_CLIP            = -250.0
RATE_VOLATILITY_WEIGHT = 1.5
SOFT_LANDING_WEIGHT    = 1.0
SOFT_LANDING_SIGMA     = 0.5

# Env
LLM_DIM   = 5
MAX_STEPS = 120

# Training
DEFAULT_EPISODES      = 500
DEFAULT_BASE_EPISODES = 10000
N_ENVS            = 4
CHECKPOINT_FREQ   = 100_000
EVAL_SEEDS        = 20

# Defaults
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_OUT   = "runs/"
DEFAULT_SEED  = 42
PPO_DEVICE    = "cuda"

# Environment variation
SHOCK_SCALE_MIN  = 0.4
SHOCK_SCALE_MAX  = 1.6
P_NO_SHOCK       = 0.30
INIT_STATE_NOISE = 0.5

# Offline state-keyed DB
DEFAULT_STATE_DB_PATH = "data/state_belief_db.json"
CHECKPOINT_EVERY_KEYS = 10
