#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/17/2016 7:26 PM
@author = Rongcheng
"""
from data import *
import numpy as np
from evaluation import *
from sklearn.neighbors import NearestNeighbors
from math import pi, cos, sin


def temporal_and_spatial(record):
    x_scale = 72
    y_scale = 150
    accuracy_scale = 5
    day_hour_scale = 2
    day_scale = 0.8
    month_scale = 0.2
    dayofyear_scale = 0.9
    dayofmonth_scale = 0.08
    year_scale = 0.15
    time = parse_time(record[3])
    day_hour = time[2]
    weekday = time[3]
    month = time[4]
    dayofyear = time[5]
    year = time[6]
    dayofmonth = time[7]
    return [record[0]*x_scale, record[1]*y_scale, accuracy_scale*np.log10(record[2] + 10), month_scale* month, year_scale * year,
           day_hour_scale*cos(day_hour*2*pi/24), day_hour_scale*sin(day_hour*2*pi/24),
           day_scale*cos(weekday*2*pi/7), day_scale*sin(weekday*2*pi/7),
           dayofmonth_scale*cos(dayofmonth*2*pi/30), dayofmonth_scale*sin(dayofmonth*2*pi/30),
           dayofyear_scale*cos(dayofyear * 2 *pi /365), dayofyear_scale*sin(dayofyear*2*pi/365)
           ]


def var_k(record):
    return int(40 + 80 * (record[3] - Setting.LASTTIME)/400000.)


class Neighbor:

    train_path = Setting.train_path
    test_path = Setting.test_path

    def __init__(self, n_neighbors=1000, algorithm="ball_tree", metric=None):
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric="manhattan")
        self.places = None
        self.data = None
        self.metric = metric

    def fit(self, data):
        self.nbrs.fit(data)

    def read_and_construct(self):
        print "reading data and indexing neareast neigbors..."
        self.places = []
        data = []
        for record in read_record(self.train_path):
            point = self.measure(record)
            data.append(point)
            self.places.append(record[-1])
        self.data = np.array(data)
        self.nbrs.fit(self.data)

    def find_nearest(self, point):
        distances, indices = self.nbrs.kneighbors(np.array([point]))
        indices = indices[0]
        distances = distances[0]
        nearest = []
        for n in range(len(indices)):
            ind = indices[n]
            nearest.append((self.places[ind], distances[n]))
        return nearest

    def get_candidates(self, record, top=5, var_k=var_k):
        k = var_k(record)
        return self.voting_nearest(self.measure(record), k, top)

    def voting_nearest(self, point, k=100, top=10):
        nearest = self.find_nearest(point)
        near_dic = {}
        for n in range(min(k, len(nearest))):
            place = nearest[n][0]
            dis = nearest[n][1]
            weight = np.exp(-0.5*dis**2/10.78)
            if place in near_dic:
                near_dic[place] += weight
            else:
                near_dic[place] = weight
        near_list = [(key, value) for key, value in near_dic.iteritems()]
        near_list.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in near_list[:top]], [x[1] for x in near_list[:top]]

    def measure(self, record):
        if self.metric is None:
            return [record[0], record[1]]
        else:
            return self.metric(record)

    def evaluate(self, top=25, k=500, var_k=None, silent=True):
        print "predicting using nearest neighbors..."
        label = []
        pred_list = []
        count = 0
        for record in read_record(self.test_path):
            if var_k is not None:
                k = var_k(record)
            pred,_ = self.voting_nearest(self.measure(record), top=top, k=k)
            pred_list.append(pred)
            label.append(record[-1])
            count += 1
            if count % 1000 == 0 and not silent:
                print count, "  recall at ", top, " is: ", recall([x[:top] for x in pred_list], label), " map3 is: ", map_3(pred_list, label)
        if not silent:
            print count, "  recall at ", top, " is: ", recall([x[:top] for x in pred_list], label), " map3 is: ", map_3(pred_list, label)
        return map_3(pred_list, label)


if __name__ == "__main__":
    neighbor = Neighbor(metric=temporal_and_spatial)
    neighbor.read_and_construct()
    neighbor.evaluate(top=25, k=500, var_k=None, silent=False)

