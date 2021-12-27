import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
data_path = "../output/dataset/census/"
x_train = pd.read_csv(data_path + '/x_train.csv')
print(x_train.loc[x_train['hours_per_week'] == 78].values[0])


def replace():
    x = x_train.loc[(x_train['age'] == 17)].iloc[0]
    x['education_num'] = 1
    x['education'] = 17
    return x.to_numpy()


x = replace()
print(x)
in_path = "../output/dataset/census/"
model = load('../output/model/census/clf_autoencoder_2.joblib')
temp_numpy = replace()

y_test_scores = model.predict(np.expand_dims(temp_numpy, 0))
print(y_test_scores)
