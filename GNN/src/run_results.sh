#!/bin/bash

# SF2H
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data SF2H --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data SF2H --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data SF2H --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data SF2H --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data SF2H --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data SF2H --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data SF2H --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single


# HighSchool
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data HighSchool --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data HighSchool --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data HighSchool --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data HighSchool --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data HighSchool --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data HighSchool --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data HighSchool --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single#

# IA-ENRON-EMPLOYEES
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
ython3 src/main.py --data ia-enron-employees --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-enron-employees --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-enron-employees --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-enron-employees --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-enron-employees --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single

# IA-CONTACT
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphConv --epoch 1500 --timestep 1 --lr 0.01 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 1  --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-contact --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-contact --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contact --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits

python3 src/main.py --data ia-contact --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contact --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contact --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contact --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 1 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single

# IA-CONTACT-HYPERTEXT2009
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits#

python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 20  --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 20  --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 20  --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg_simp --model GraphSage --epoch 1500 --lr 0.01 --timestep 20  --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits#

python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphConv --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits#

python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct agg --model GraphSage --epoch 1500 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits#

python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 20 --metric auc --test_agg True --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges False --predictor dotProduct --loss_func BCEWithLogits --step_prediction single
python3 src/main.py --data ia-contacts_hypertext2009 --feat_struct temporal_edges --model GraphSage --epoch 700 --lr 0.01 --timestep 20 --metric auc --test_agg False --duplicate_edges True --predictor dotProduct --loss_func BCEWithLogits --step_prediction single