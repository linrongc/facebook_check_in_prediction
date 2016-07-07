#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@time = 6/17/2016 10:38 AM
@author = Rongcheng
"""
from setting import *
import cPickle


def parse_record(fields):
    if len(fields) == 6:
        return [float(fields[1]), float(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])]
    elif len(fields) == 5:
        return [float(fields[1]), float(fields[2]), int(fields[3]), int(fields[4])]
    else:
        raise Exception("parse error")


def no_parse(fields):
    return fields


def read_record(path, parser=parse_record, skip=-1):
    print "reading records from ", path
    count = 0
    with open(path, "r") as f:
        title = f.next()
        if skip != -1:
            for n in range(skip):
                f.next()
        for line in f:
            fields = line.rstrip("\n").split(",")
            yield parser(fields)
            count += 1
            if count % 1000000 == 0:
                print count,
    print "reading is done!"


def parse_time(time):
    '''
    :param time: in minute
    :return: hour day day_hour weekday month
    '''
    hour = time/60.
    day = hour/24
    day_hour = hour % 24
    weekday = day % 7
    dayofyear = day % 365
    dayofmonth = day%30
    week = day/7
    month = day/30
    year = day/365.0
    return [hour, day, day_hour, weekday, month, dayofyear, year, dayofmonth, week]


def dump_data(data, path):
    print "dump data to ", path
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
    print "dump complete!"


def load_data(path):
    print "load data from ", path
    with open(path, "rb") as f:
        data = cPickle.load(f)
        print "load complete!"
        return data


if __name__ == "__main__":
    read_record(Setting.train_path)