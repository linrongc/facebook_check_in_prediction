#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/17/2016 10:38 AM
@author = Rongcheng
"""


class Setting:
    MODE = "validate"
    RERUN = False
    result_path = "../result/"

    if MODE == "validate":
        LASTTIME = 590400
        grid_path = "../data/time_split/grid/"
        train_path = "../data/time_split/train.csv"
        test_path = "../data/time_split/test.csv"

    elif MODE == "submit":
        LASTTIME = 786240
        grid_path = "../data/submit/grid/"
        train_path = "../data/train.csv"
        test_path = "../data/test.csv"