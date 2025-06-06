model=$1
chkpt=$2
dir=$3

cd $dir
python tools/deployment/export_onnx.py --check -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml -r $chkpt
