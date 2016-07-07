#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/17/2016 10:54 AM
@author = Rongcheng
"""


def map_3(pred, label):
    score = 0.0
    n_record = 0
    for rank, actual in zip(pred, label):
        if actual in rank:
            for n in range(3):
                if rank[n] == actual:
                    score += 1.0 / (n + 1)
                    break
        n_record += 1

    return score / n_record


def recall(pred, label):
    score  = 0.0
    n_record = 0
    for pred_set, actual in zip(pred, label):
        if actual in pred_set:
            score += 1
        n_record += 1
    return score / n_record
