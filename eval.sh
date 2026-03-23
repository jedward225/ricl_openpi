cd /home/ruoqu/jjliu/RoboRetry/aha_ricl

export DISPLAY=:0
export COPPELIASIM_ROOT=/home/ruoqu/jjliu/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

HF_HUB_OFFLINE=1 uv run python scripts/eval_ricl_rlbench.py \
      --checkpoint /home/ruoqu/jjliu/shared/trained_models/ricl_no_interporation \
      --demos_dir ./processed_rlbench_25 \
      --task all --episodes 25

# uv run python scripts/eval_ricl_rlbench.py \
#     --checkpoint /home/ruoqu/jjliu/shared/trained_models/yingxi/n50/rlbench_ricl_50/30000 \
#     --demos_dir ./processed_rlbench_25 \
#     --task close_box_hard,insert_onto_square_peg,pick_and_lift,pick_up_cup,push_button,slide_block_to_target,straighten_rope,turn_tap \
#     --episodes 25 \
#     --headless 2>&1 | tee eval_ricl_all_tasks.log


