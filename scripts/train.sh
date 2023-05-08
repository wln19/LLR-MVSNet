#!/usr/bin/env bash
MVS_TRAINING="/media/yons/10T1/wanglina/cas_training/dtu"
LOG_DIR="./checkpoints"
export NGPUS=1
if [ ! -d $LOG_DIR ]; then
	    mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch  --nproc_per_node=1 train.py --logdir="./checkpoints" --dataset=dtu_yao  --trainpath=$MVS_TRAINING \
	                --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 4 --eval_freq 3    \
			                --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt   \
							--cost_aggregation 91 \
							--resume \
					        --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt  \
                                        