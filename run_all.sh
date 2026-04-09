DATA_ROOT=${1:-./data/ACDC}
CHECKPOINT=${2:-./checkpoints/sam_vit_b_01ec64.pth}

python run.py \
  --phase all \
  --data_root $DATA_ROOT \
  --sam_checkpoint $CHECKPOINT