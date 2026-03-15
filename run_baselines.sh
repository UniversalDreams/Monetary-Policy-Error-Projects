#!/bin/bash

set -e

echo "==================================================="
echo "STARTING DRQN FULL RUN"
echo "==================================================="
python src/train_custom.py \
    --policy drqn \
    --condition all \
    --base-episodes 20000 \
    --run-name drqn_final_run

echo "==================================================="
echo "STARTING MLP FULL RUN"
echo "==================================================="
python src/train_custom.py \
    --policy mlp \
    --condition all \
    --base-episodes 20000 \
    --run-name mlp_final_run

echo "==================================================="
echo "STARTING LSTM FULL RUN"
echo "==================================================="
python src/train_custom.py \
    --policy lstm \
    --condition all \
    --base-episodes 20000 \
    --run-name lstm_final_run

echo "==================================================="
echo "ALL TRAINING COMPLETE!"
echo "==================================================="