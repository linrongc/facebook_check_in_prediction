#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/18/2016 11:24 AM
@author = Rongcheng
"""

from data import *
from evaluation import *
import numpy as np
import os


def grid_cut(x, y, xsplit, ysplit):
    if x >= 10:
        x = 10 - 1e-5
    if x < 0:
        x = 0
    if y >= 10:
        y = 10 - 1e-5
    if y < 0:
        y = 0
    return int(x/10*xsplit), int(y/10*ysplit)


class Grid:
    train_path = Setting.train_path
    test_path = Setting.test_path

    def __init__(self, xsplit=10, ysplit=10):
        self.grid_place = []
        self.xsplit = xsplit
        self.ysplit = ysplit

    def split_places(self):
        print "split places into grids..."
        for m in range(self.xsplit):
            for n in range(self.ysplit):
                place_set = set()
                self.grid_place.append(place_set)
        for record in read_record(self.train_path):
            x_ind, y_ind = grid_cut(record[0], record[1], self.xsplit, self.ysplit)
            slot = x_ind*self.ysplit + y_ind
            place_id = record[-1]
            try:
                if place_id not in self.grid_place[slot]:
                    self.grid_place[slot].add(place_id)
            except IndexError:
                print record
                exit(0)
        grid_size = np.array([len(x) for x in self.grid_place])
        print "min = ", grid_size.min(), " max = ", grid_size.max(), " average = ", grid_size.sum()/len(grid_size)

    def grid_recall(self):
        """
        :return: the recall rate in grid
        """

        label = []
        pred_list = []
        count = 0
        for record in read_record(self.test_path):
            x_ind, y_ind = grid_cut(record[0], record[1], self.xsplit, self.ysplit)
            place_id = record[-1]
            slot = x_ind*self.ysplit + y_ind
            label.append(place_id)
            pred_list.append(self.grid_place[slot])
            count += 1
            if count % 10000 == 0:
                print "at ", count, " recall is: ", recall(pred_list, label)
        print self.xsplit, " X ", self.ysplit, " grid recall: ", recall(pred_list, label)

    def split_file(self):
        """
        split training and testing file into grid folders
        """
        title = "row_id,x,y,accuracy,time,place_id\n"
        print "splitting files into grid files..."
        sub_folder = os.path.join(Setting.grid_path, str(self.xsplit)+"_"+str(self.ysplit))
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        for m in range(self.xsplit):
            # to avoid open too many files (ysplit should less than 1000 here)
            print "starting No.", m, " subprocess..."
            train_writers = []
            for n in range(self.ysplit):
                xfolder = os.path.join(sub_folder, str(m))
                if not os.path.exists(xfolder):
                    os.mkdir(xfolder)
                yfolder = os.path.join(xfolder, str(n))
                if not os.path.exists(yfolder):
                    os.mkdir(yfolder)
                train_file = os.path.join(yfolder, "train.csv")
                train_writers.append(open(train_file, "w"))
                train_writers[-1].write(title)

            for record in read_record(self.train_path):
                place_id = record[-1]
                rec_str = ",".join([str(x) for x in record])
                for n in range(self.ysplit):
                    row_id = 1
                    slot = m*self.ysplit + n
                    if place_id in self.grid_place[slot]:
                        train_writers[n].write(str(row_id) + "," + rec_str + "\n")
                        row_id += 1

            for writer in train_writers:
                writer.close()

            test_writers = []
            for n in range(self.ysplit):
                test_file = os.path.join(sub_folder, str(m), str(n), "test.csv")
                test_writers.append(open(test_file, "w"))
                test_writers[-1].write(title)

            for record in read_record(self.test_path):
                x_ind, y_ind = grid_cut(record[0], record[1], self.xsplit, self.ysplit)
                grid_slot = x_ind*self.ysplit + y_ind
                for n in range(self.ysplit):
                    row_id = 1
                    slot = m*self.ysplit + n
                    if grid_slot == slot:
                        rec_str = ",".join([str(x) for x in record])
                        test_writers[n].write(str(row_id) + "," + rec_str + "\n")
                        row_id += 1

            for writer in test_writers:
                writer.close()


if __name__ == "__main__":
    grid = Grid(xsplit=5, ysplit=20)
    grid.split_places()
    grid.split_file()
    grid.grid_recall()

