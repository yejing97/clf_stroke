#!/bin/bash

conda activate stroke_clf

python3 train.py --feature_type="FG_EMB" --filter_type="los" --lr=1e-2 --epoches=1
python3 train.py --feature_type="FR_EMB" --filter_type="los" --lr=1e-2 --epoches=1
python3 train.py --feature_type="FG_EMB" --filter_type="rel" --lr=1e-2 --epoches=1
python3 train.py --feature_type="FR_EMB" --filter_type="rel" --lr=1e-2 --epoches=1