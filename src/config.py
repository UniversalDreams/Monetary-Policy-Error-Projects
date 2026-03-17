# src/config.py  — single source of truth for all hyperparameters

# PPO (MLP)
LR         = 5e-4
N_STEPS    = 240
BATCH_SIZE = 60
N_EPOCHS   = 4
GAMMA      = 0.99

# LSTM (shared structural params)
LSTM_N_STEPS     = 512
LSTM_BATCH_SIZE  = 128
LSTM_N_EPOCHS        = 4
LSTM_N_CRITIC_EPOCHS = 10     # critic trains 10 epochs vs 4 for actor
LSTM_CRITIC_LR       = 1e-3   # critic LR 5x actor LR (2e-4) to converge faster
LSTM_HIDDEN_SIZE     = 256

# Baseline condition
BASELINE_LR             = 3e-4
BASELINE_LR_END         = 5e-5
BASELINE_LR_DECAY_START = 0.6
BASELINE_ENT_COEF       = 0.005
BASELINE_ENT_COEF_END   = 0.001
BASELINE_CLIP_RANGE     = 0.10
BASELINE_VF_COEF        = 0.5

# Oracle condition
ORACLE_LR             = 3e-4
ORACLE_LR_END         = 1e-5
ORACLE_LR_DECAY_START = 0.6
ORACLE_ENT_COEF       = 0.005
ORACLE_ENT_COEF_END   = 0.001
ORACLE_CLIP_RANGE     = 0.10
ORACLE_VF_COEF        = 0.5

# LLM offline condition
LLM_LR             = 1e-4
LLM_LR_END         = 1e-5
LLM_LR_DECAY_START = 0.6
LLM_ENT_COEF       = 0.01
LLM_ENT_COEF_END   = 0.002
LLM_CLIP_RANGE     = 0.10
LLM_VF_COEF        = 0.05

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
DEFAULT_BASE_EPISODES = 20000
N_ENVS            = 4
CHECKPOINT_FREQ   = 100_000
EVAL_SEEDS        = 20

# Defaults
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_OUT   = "runs_custom/"
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
