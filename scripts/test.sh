#!/usr/bin/env bash
# TESTPATH="/root/autodl-tmp/autodl-tmp/tank/intermediate"
# TESTLIST="/root/autodl-tmp/autodl-tmp/tank/intermediate/scan_list_test.txt"
# TESTPATH="/media/yons/10T1/wanglina/tank/intermediate"
# TESTLIST="lists/tank/test.txt"
# TESTPATH="/media/yons/10T1/wanglina/dtu/dtu"
# TESTLIST="lists/dtu/test.txt"
TESTPATH=/media/yons/10T1/wanglina/eth3d_high_res_test
TESTLIST="lists/dtu/test.txt"
# TESTPATH="/media/yons/10T1/wanglina/campus/L1/text"
# TESTLIST="lists/dtu/test.txt"
export save_results_dir="./outputs"
CKPT_FILE="/media/yons/10T1/wanglina/casmvsnetfse/CasMVSNet/checkpoints1/model_000029.ckpt"
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST \
                          --outdir $save_results_dir  --interval_scale 1.06   \
                          --filter_method gipuma \
                          --prob_threshold 0.3 \
                          --disp_threshold 0.25 \
                          --num_consistent 1 \
                          --loadckpt $CKPT_FILE ${@:2}

