from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import json

parent_path = '../output/anomaly_included'
dataset_name = 'census'
anomaly_type = 'distance_sex_hpw_workclass'
full_path = f'{parent_path}/{dataset_name}/{anomaly_type}/explanations'
file_limit = 100

color = list(list(np.random.choice(range(256), size=3)) for _ in range(10))
color = ['lightcoral', 'indianred', 'brown', 'maroon', 'mediumpurple', 'plum', 'midnightblue', 'mediumblue',
         'slateblue']
for file_num in range(file_limit):
    json_file_path = full_path + '/' + str(file_num) + '.json'
    with open(json_file_path, 'r') as file:
        json_file = json.load(file)
        feature_names = list(json_file.keys())
        feature_values = list(json_file.values())
        feature_values.reverse()
        feature_names.reverse()
        plt.barh(y=feature_names, width=feature_values, color=color)
        plt.savefig(full_path + '/' + str(file_num) + '.png')
        plt.clf()
