# -*- coding: utf-8 -*-

import argparse
from preprocessing import temporal_graph
from utils import train_test_split

parser = argparse.ArgumentParser('Preprocessing data')
parser.add_argument('--data', type=str, help='Dataset name : \{SF2H\}', default='SF2H')
args = parser.parse_args()

# Temporal graph
g = temporal_graph(args.data)

print('SF2H graph')
print(g)

# training graph
'''
VAL_SIZE = 0.15
TEST_SIZE = 0.15

train_g, train_pos_g, train_neg_g, \
    val_pos_g, val_neg_g, \
    test_pos_g, test_neg_g = train_test_split(g, VAL_SIZE, TEST_SIZE)

print('Training graph')
print(train_g)


print('train_pos_g', train_pos_g)
print('train_neg_g', train_neg_g)
print('val_pos_g', val_pos_g)
print('val_neg_g', val_neg_g)
print('test_pos_g', test_pos_g)
print('test_neg_g', test_neg_g)'''
