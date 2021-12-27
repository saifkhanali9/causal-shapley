import numpy as np
import shap
import pandas as pd
import pickle

from joblib import load


def main(file_name='synthetic_discrete_3', xi=0):
    file_name = 'census/x_train.csv'
    # file_name = 'synthetic_discrete_3.csv'
    file_path = '../output/dataset/' + file_name
    model_name = 'census/xgb_clf.pkl'
    # model_name = 'synthetic_discrete_3.sav'
    model_path = '../output/model/' + model_name
    df = pd.read_csv(file_path)
    feature_names = df.columns
    try:
        model = pickle.load(open(model_path, 'rb'))
    except pickle.UnpicklingError:
        model = load(model_path)
    # n_features = len(df.columns[:-1])
    X = df.to_numpy()
    # X = shap.utils.sample(X, 100)
    # explainer = shap.Explainer(model.predict)
    # shap_values = explainer(X)
    print("Model coefficients:\n")
    # for i in range(X.shape[1]):
    #     print(X.columns[i], "=", model.coef_[0][i].round(4))
    explainer = shap.Explainer(model.predict, X)
    x_temp = pd.DataFrame(data=np.array([[31, 57, 1, 1, 44, 49, 19, 80]]), columns=feature_names)
    x_temp = pd.DataFrame(data=np.array([X[15]]), columns=feature_names)
    shap_values = explainer(x_temp)
    # visualize the first prediction's explanation
    shap.waterfall_plot(shap_values[0])


main()
