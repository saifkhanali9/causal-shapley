import model_wrapper
import numpy as np

from environment import pytorch_environment

x = np.array([0, 0, 1, 1])
with pytorch_environment():
    clf = model_wrapper.ScikitLearnClassificationModel('1st','synthetic_data.sav')
    print(clf.predict())