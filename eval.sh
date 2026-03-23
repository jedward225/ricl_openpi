#!/bin/bash
# RICL Evaluation — 2 versions × within-task retrieval × 100 eps
# Run on local 5090

cd /home/ruoqu/jjliu/RoboRetry/aha_ricl

export DISPLAY=:0
export COPPELIASIM_ROOT=/home/ruoqu/jjliu/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

TASKS="close_box_hard,pick_and_lift,pick_up_cup,push_button,slide_block_to_target,turn_tap"

echo "============================================"
echo "1/2: RICL with-interpolation (within-task)"
echo "     Checkpoint: ricl_rlbench/19999"
echo "============================================"
HF_HUB_OFFLINE=1 uv run python scripts/eval_ricl_rlbench.py \
    --checkpoint /home/ruoqu/jjliu/shared/trained_models/ricl_rlbench/19999 \
    --demos_dir ./processed_rlbench_25 \
    --task "$TASKS" --episodes 100

echo ""
echo "============================================"
echo "2/2: RICL no-interpolation (within-task)"
echo "     Checkpoint: ricl_no_interporation"
echo "============================================"
HF_HUB_OFFLINE=1 uv run python scripts/eval_ricl_rlbench.py \
    --checkpoint /home/ruoqu/jjliu/shared/trained_models/ricl_no_interporation \
    --demos_dir ./processed_rlbench_25 \
    --task "$TASKS" --episodes 100 \
    --no_interpolation
