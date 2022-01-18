import json
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parent_path = '../output/anomaly_included/'
dataset_name = 'census/'

anomaly_types = ["distance_age_education_sex",
                 "distance_age_hpw_sex", "distance_sex_hpw_workclass", "jp_hpw_age_sex",
                 "jp_sex_hpw_education_age", "jp_workclass_age_education", "tuner_hpw_age"]


def build_dict(feature_list, explanation_path):
    file_no = 1
    main_dict = {name: [] for name in feature_list}
    while True:
        file = explanation_path + str(file_no) + '.json'
        if not path.exists(file):
            break
        with open(file, 'r') as f:
            explanation = json.load(f)
            for key in explanation.keys():
                main_dict[key].append(explanation[key])
        file_no += 1
    return main_dict


def mean_std():
    for anomaly_type in anomaly_types:
        print(anomaly_type)
        # Just to get csv
        dataset = pd.read_csv(f'{parent_path}{dataset_name}{anomaly_type}/anomalous_data.csv')
        feature_names = list(dataset.columns)
        till_anomalytype = f'{parent_path}{dataset_name}{anomaly_type}/'
        explanation_path = f'{parent_path}{dataset_name}{anomaly_type}/explanations/'
        main_dict = build_dict(feature_list=feature_names, explanation_path=explanation_path)
        mean_dict = {name: np.mean(main_dict[name]) for name in feature_names}
        std_dict = {name: np.std(main_dict[name]) for name in feature_names}

        # sort
        mean_dict_sorted = dict(sorted(mean_dict.items(), key=lambda item: item[1]))
        std_dict_sorted = dict(sorted(std_dict.items(), key=lambda item: item[1]))

        with open(till_anomalytype + 'explanation_mean.json', 'w') as file:
            json.dump(mean_dict_sorted, file)
        with open(till_anomalytype + 'explanation_std.json', 'w') as file:
            json.dump(std_dict_sorted, file)

        # saving plots
        sns.barplot(y=list(mean_dict_sorted.keys()), x=list(mean_dict_sorted.values()))
        plt.savefig(till_anomalytype + 'explanation_mean.png')
        plt.clf()

        sns.barplot(y=list(std_dict_sorted.keys()), x=list(std_dict_sorted.values()))
        plt.savefig(till_anomalytype + 'explanation_std.png')
        plt.clf()
    print("\nDONE!")


def box_plots():
    features = list(pd.read_csv(f'{parent_path}{dataset_name}{anomaly_types[0]}/x_train.csv').columns)
    for anomaly_type in anomaly_types:
        print(anomaly_type)
        till_anomalytype = f'{parent_path}{dataset_name}{anomaly_type}/'
        explanation_path = f'{parent_path}{dataset_name}{anomaly_type}/explanations/'
        main_dict = build_dict(feature_list=features, explanation_path=explanation_path)

        red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

        fig, axs = plt.subplots(1, len(features), figsize=(20, 10))

        for i, ax in enumerate(axs.flat):
            ax.boxplot(main_dict[list(main_dict.keys())[i]], flierprops=red_circle)
            ax.set_title(list(main_dict.keys())[i], fontsize=20, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)

        # plt.tight_layout()
        plt.show()
        # plt.savefig(till_anomalytype + 'box_plots.png')
        # plt.clf()
        break


box_plots()
