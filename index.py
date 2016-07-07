#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/19/2016 12:20 PM
@author = Rongcheng
"""
from data import *


class Index:

    train_path = Setting.train_path

    def __init__(self):
        self.records = []
        self.place_index = {}

    def construct_index(self):
        for record in read_record(self.train_path):
            place_id = record[-1]
            self.records.append(record)
            if place_id not in self.place_index:
                self.place_index[place_id] = []
            self.place_index[place_id].append(record)
