import shap
import pandas as pd
import pickle


def main(file_name='synthetic2', xi=0):
    df = pd.read_csv('../output/dataset/' + file_name + '.csv')
    print(df.columns)
    model = pickle.load(open('../output/model/' + file_name + '.sav', 'rb'))
    n_features = len(df.columns[:-1])
    X = df.iloc[:, :n_features]
    # X = shap.utils.sample(X, 100)
    # explainer = shap.Explainer(model.predict)
    # shap_values = explainer(X)
    print("Model coefficients:\n")
    for i in range(X.shape[1]):
        print(X.columns[i], "=", model.coef_[i].round(4))
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    # visualize the first prediction's explanation
    shap.waterfall_plot(shap_values[0])


main()
