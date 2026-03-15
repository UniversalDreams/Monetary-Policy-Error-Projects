#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==================================================="
echo "🚀 STARTING MLP (STANDARD PPO) FULL TRAINING RUN"
echo "==================================================="
python src/train_custom.py \
    --policy mlp \
    --condition all \
    --base-episodes 20000 \
    --run-name mlp_final_run

echo "==================================================="
echo "🚀 STARTING LSTM (RECURRENT PPO) FULL TRAINING RUN"
echo "==================================================="
python src/train_custom.py \
    --policy lstm \
    --condition all \
    --base-episodes 20000 \
    --run-name lstm_final_run

echo "==================================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "==================================================="