#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

def parse_num_from_str(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def get_metadata_from_log(data_set_num):
    model_hyperparameters_filename = "./log/classifier_log" + str(data_set_num) + "/AutoML(1):simulated" + str(data_set_num) + ".log"
    print("Parsing metadata/metafeatures from " + model_hyperparameters_filename)
    metafeatures_filename = "./log/classifier_log" + str(data_set_num) + "/metafeatures.json"
    with open(model_hyperparameters_filename) as f:
        parse_line = False
        count = 0
        metafeatures = {}
        for line in f:
            if not parse_line and 'Metafeatures for dataset' in line and '[INFO]' in line:
                count += 1
                parse_line = True
            elif parse_line:
                l = line.split(':')
                if len(l) == 2:
                    metafeatures[l[0].strip()] = parse_num_from_str(l[1].strip().replace('\n', ''))
                else:
                    #print(metafeatures)
                    break
        with open(metafeatures_filename, 'w') as fp:
            json.dump(metafeatures, fp)
                    