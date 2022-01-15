import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parent_path = '../output/anomaly_included/'
dataset_name = 'census/'

anomaly_types = ["distance_age_educationnum_sex",
                 "distance_age_hpw_sex", "distance_sex_hpw_workclass", "jp_hpw_age_sex", "jp_hpw_educationnum_age_sex",
                 "jp_sex_hpw_education_age", "jp_workclass_age_education", "tuner_hpw_age"]

for anomaly_type in anomaly_types:
    print(anomaly_type)
    # Just to get csv
    dataset = pd.read_csv(f'{parent_path}{dataset_name}{anomaly_type}/anomalous_data.csv')
    feature_names = list(dataset.columns)
    main_dict = {name: [] for name in feature_names}
    till_anomalytype = f'{parent_path}{dataset_name}{anomaly_type}/'
    explanation_path = f'{parent_path}{dataset_name}{anomaly_type}/explanations/'
    files = [explanation_path + str(i) + '.json' for i in range(1, 101)]
    for num, file in enumerate(files):
        with open(file, 'r') as f:
            explanation = json.load(f)
            for key in explanation.keys():
                main_dict[key].append(explanation[key])
            # all_explanations[num] = np.array(list(explanation.values()))

    # for mean
    # for key in main_dict.keys():
    #     main_dict[key] = np.mean(main_dict[key])
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
    plot = sns.barplot(y=list(mean_dict_sorted.keys()), x=list(mean_dict_sorted.values()))
    plt.savefig(till_anomalytype + 'explanation_mean.png')
    plt.clf()

    plot = sns.barplot(y=list(std_dict_sorted.keys()), x=list(std_dict_sorted.values()))
    plt.savefig(till_anomalytype + 'explanation_std.png')
    plt.clf()
print("\nDONE!")
