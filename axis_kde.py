#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 7/5/2016 12:32 AM
@author = Rongcheng
"""
from sklearn.neighbors import KernelDensity
import numpy as np
from index import Index
from neareast_neighbor import Neighbor, var_k, temporal_and_spatial
from data import *
from evaluation import *
import os, sys
from multi_thread import MultiThread
import time, random
from subprocess import Popen
import matplotlib.pyplot as plt


def load_neighbor(neighbor_path, train_path, test_path=None, metric=temporal_and_spatial):
    print "load or construct nearest neighbor..."
    if not Setting.RERUN and os.path.exists(neighbor_path):
        neighbor = load_data(neighbor_path)
    else:
        neighbor = Neighbor(metric=metric)
        neighbor.train_path = train_path
        neighbor.test_path = test_path
        neighbor.read_and_construct()
        dump_data(neighbor, neighbor_path)
    print "done!"
    return neighbor


def load_index(index_path, train_path, reconstruct=Setting.RERUN):
    print "load or construct index..."
    if not reconstruct and os.path.exists(index_path):
        index = load_data(index_path)
    else:
        index = Index()
        index.train_path = train_path
        index.construct_index()
        dump_data(index, index_path)
    print "done!"
    return index


ftr_weight = [30, 200, 5, 1, 20, 5, 0.2, 0.1, 1]  # convert the density to a reasonable scale


def den_ftr(record):
    time = parse_time(record[3])
    day_hour = time[2]
    week_day = time[3]
    year = time[6]
    day_of_month = time[7]
    day_of_year = time[5]
    return [record[0]*ftr_weight[0], record[1]*ftr_weight[1], np.log10(record[2] + 5)*ftr_weight[2],
            day_hour*ftr_weight[3], week_day*ftr_weight[4], year*ftr_weight[5], day_of_month*ftr_weight[6],
            day_of_year * ftr_weight[7]]


def list_den_ftr(record_list):
    den_rec_list = []
    for record in record_list:
        den_rec_list.append(den_ftr(record))
    return den_rec_list


class AxisDensity:
    # [node_num, neighbor_prob], x, y, accuracy, day_hour, week_day, year, month_day, day_of_year
    band_width = [0.3, 0.72, 0.7, 0.5, 1.07, 2.3, 0.7, 3.0]
    weights = [0.88, 3.0, 1.4, 1.9, 0.9, 1.0, 1.5, 5.5, 0.5, 0.25]
    kernel = ["gaussian", "gaussian", "cosine", "gaussian", "gaussian", "exponential", "gaussian", "gaussian"]
    smooth_factors = [50, 50, 50, 50, 80, 80, 80, 50]

    def __init__(self, index, neighbor):
        self.place_kde = {}
        self.place_size = {}
        self.index = index
        self.neighbor = neighbor

    def smooth_func(self, size, factor):
        return np.log(factor)/np.log(size + 1)

    def build_kdes(self):
        for place_id, record_list in self.index.place_index.iteritems():
            kdes = []
            den_rec_list = list_den_ftr(record_list)
            ftr_list = zip(*den_rec_list)
            for n in range(len(ftr_list)):
                kdes.append(self.fit_kde(ftr_list[n], self.band_width[n] * self.smooth_func(len(record_list), self.smooth_factors[n]), self.kernel[n]))
            self.place_kde[place_id] = kdes
            self.place_size[place_id] = np.log(len(record_list))

    def fit_kde(self, attr, band_width, kernel):
        return KernelDensity(bandwidth=band_width, kernel=kernel).fit(np.array(attr).reshape((-1, 1)))

    def log_proba(self, kde, value, period=None):
        if period is None:
            return kde.score([[value]])
        else:
            values = kde.score_samples([[value], [value+period], [value-period]])
            total = np.logaddexp(values[0], values[1])
            total = np.logaddexp(total, values[2])
            return total

    def score(self, record, place_id, knn_weight=1):
        value = [self.weights[0] * self.place_size[place_id], self.weights[1]*np.log(knn_weight)]
        kdes = self.place_kde[place_id]
        den_rec = den_ftr(record)
        for n in range(len(kdes)):
            if n == 3:
                period = 24 * ftr_weight[3]
            elif n == 4:
                period = 7 * ftr_weight[4]
            elif n == 6:
                period = 30 * ftr_weight[6]
            elif n == 7:
                period = 365. * ftr_weight[7]
            elif n == 8:
                period = 4 * ftr_weight[7]
            else:
                period = None
            log_pro = self.log_proba(kdes[n], den_rec[n], period)
            log_pro = np.logaddexp(log_pro, np.log(0.001))
            value.append(self.weights[n+2] * log_pro)
        return sum(value)

    def evaluate(self, test_path, top=3, num_cand=10):
        label_list = []
        pred_list = []
        count = 0
        print "num_cand = ", num_cand
        print "feature weight = ", ftr_weight
        print "smooth = ", self.smooth_factors
        for record in read_record(test_path):
            k = int(1.15*var_k(record))
            candidates, weight_list = self.neighbor.voting_nearest(self.neighbor.measure(record), top=num_cand, k=k)
            score_list = []
            total_weight = sum(weight_list)
            for cand, weight in zip(candidates, weight_list):
                score_list.append((cand, self.score(record, cand, weight/total_weight)))
            score_list.sort(key=lambda x: x[1], reverse=True)
            label_list.append(record[-1])
            pred_list.append([x[0] for x in score_list[:top]])
            count += 1
            if count%1000 == 0:
                print count, " map3 = ", map_3(pred_list, label_list)
        print count, " map3 = ", map_3(pred_list, label_list)

    def predict(self, folder):
        test_path = os.path.join(folder, "test.csv")
        pred_list = []
        weight_list = []
        for record in read_record(test_path):
            k = int(1.15*var_k(record))
            candidates, weights = self.neighbor.voting_nearest(self.neighbor.measure(record), top=25, k=k)
            score_list = []
            total_weight = sum(weights)
            for cand, weight in zip(candidates, weights):
                score_list.append((cand, self.score(record, cand, weight/total_weight)))
            score_list.sort(key=lambda x: x[1], reverse=True)
            pred_list.append([x[0] for x in score_list])
            weight_list.append([x[1] for x in score_list])
        kde_path = os.path.join(folder, "kde_candidates")
        dump_data([pred_list, weight_list], kde_path)

    def draw_random_place(self, dim):
        place_id = random.choice(self.index.place_index.keys())
        record_list = self.index.place_index[place_id]
        den_rec_list = list_den_ftr(record_list)
        ftr_list = zip(*den_rec_list)
        self.draw_1d_density(ftr_list[dim],
            self.band_width[dim] * self.smooth_func(len(record_list), self.smooth_factors[dim]), self.kernel[dim])

    @staticmethod
    def draw_1d_density(attr, bandwidth, kernel="gaussian"):
        attr = np.array(attr)[:,np.newaxis]
        kde = KernelDensity(kernel=kernel, metric="manhattan", bandwidth=bandwidth).fit(attr)
        x_plot = np.linspace(min(attr), max(attr), 1000)[:, np.newaxis]
        fig, ax = plt.subplots()
        log_dens = kde.score_samples(x_plot)
        ax.plot(x_plot, np.exp(log_dens))
        ax.plot(attr[:, 0], -0.01 - 0.02*np.random.random(attr.shape[0]), '+k')
        plt.show()

### following is the multi-thread support for computing the grid kde

def parallel_kde(grid_folder, x_start, x_end, y_start, y_end, num_thread):
    par_list = []
    for m in range(x_start, x_end + 1):
        for n in range(y_start, y_end + 1):
            sub_folder = os.path.join(grid_folder, str(m), str(n))
            par_list.append(sub_folder)
    mt = MultiThread(kde_start, par_list, max_thread=num_thread)
    mt.start()
    mt.join()


def kde_start(folder):
    time.sleep(random.random()*0.1)
    log_path = os.path.join(folder, "axe_kde.log")
    logfile = open(log_path, "w")
    command = "python axis_kde.py " + folder
    print "start process: ", folder
    process = Popen(command, stdout=logfile, shell=True)
    process.wait()
    logfile.close()
    print "end process: ", folder


def folder_kde(folder):
    train_path = os.path.join(folder, "train.csv")
    neighbor_path = os.path.join(folder, "neighbor")
    index_path = os.path.join(folder, "index")

    neighbor = load_neighbor(neighbor_path, train_path)
    index = load_index(index_path, train_path)
    axkde = AxisDensity(index, neighbor)
    axkde.build_kdes()
    axkde.predict(folder)


def test(folder):
    train_path = os.path.join(folder, "train.csv")
    test_path = os.path.join(folder, "test.csv")
    neighbor_path = os.path.join(folder, "neighbor")
    index_path = os.path.join(folder, "index")

    neighbor = load_neighbor(neighbor_path, train_path)
    index = load_index(index_path, train_path)
    axkde = AxisDensity(index, neighbor)
    axkde.build_kdes()
    axkde.draw_random_place(6)
    print "combine weight: ", axkde.weights
    print "bandwidth: ", axkde.band_width
    axkde.evaluate(test_path)
    # axkde.predict(folder)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 2:
        test("../data/time_split/5_20/1/5/")
    elif argc == 2:
        folder_kde(sys.argv[1])
    elif argc == 7:
        parallel_kde(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))