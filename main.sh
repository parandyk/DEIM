mode=$1
dataset=$2
model=$3
chkpt=$4
GPUS=$5
dir=$6

cd $dir

#CUDA_VISIBLE_DEVICES=0 

if [ "$mode" == train ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_${dataset}.yml --summary-dir --use-amp --seed=0
elif [ "$mode" == tune ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_${dataset}.yml --summary-dir --use-amp --seed=0 -t $chkpt
elif [ "$mode" == test ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_${dataset}.yml --summary-dir --test-only -r $chkpt
fi


