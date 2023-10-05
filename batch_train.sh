#!/bin/bash

# conda activate stroke_clf

python3 train_edge.py --feature_type="FG_EMB" --filter_type="los" --lr=1e-2 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="los" --lr=1e-2 --epoches=100
python3 train_edge.py --feature_type="FG_EMB" --filter_type="rel" --lr=1e-2 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="rel" --lr=1e-2 --epoches=100
python3 train_edge.py --feature_type="FG_EMB" --filter_type="los" --lr=1e-4 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="los" --lr=1e-4 --epoches=100
python3 train_edge.py --feature_type="FG_EMB" --filter_type="rel" --lr=1e-4 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="rel" --lr=1e-4 --epoches=100
python3 train_edge.py --feature_type="FG_EMB" --filter_type="los" --lr=1e-6 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="los" --lr=1e-6 --epoches=100
python3 train_edge.py --feature_type="FG_EMB" --filter_type="rel" --lr=1e-6 --epoches=100
python3 train_edge.py --feature_type="FR_EMB" --filter_type="rel" --lr=1e-6 --epoches=100