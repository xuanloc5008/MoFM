#!/bin/bash

SEEDS=${1:-5}
BASE_SEED=${2:-42}

python run.py \
  --phase all \
  --seeds $SEEDS \
  --seed $BASE_SEED