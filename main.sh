GPUS=$1
model=$2
cd $3
#CUDA_VISIBLE_DEVICES=0 
torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --use-amp --seed=0
