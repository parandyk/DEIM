mode=$1
model=$2
chkpt=$3
GPUS=$4
dir=$5

cd $dir

#CUDA_VISIBLE_DEVICES=0 

if [ "$mode" == train ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --summary-dir --use-amp --seed=0
elif [ "$mode" == tune ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --summary-dir --use-amp --seed=0 -t $chkpt
elif [ "$mode" == test ]
then
  torchrun --master_port=7777 --nproc_per_node=$GPUS train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --summary-dir --test-only -r $chkpt
fi


