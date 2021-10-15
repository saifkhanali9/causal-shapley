## Running the code:
- Just run the code by pressing play button.
- You can make configurations in you run by changing the very last line of causal_shaplye.py
    - main(version='4', file_name='synthetic_discrete_2', local_shap=15, is_classification=True, global_shap=False)
- Argument file_name has all the necessary information for the dataset. A csv file is located under output/dataset/file_name.csv which contains the complete dataset to be used for Shapley value computation.While causal structure of the data is located under  which is located under output/dataset/file_name/causal_struct.json
- Argument version specifies which version of shapley value you want to run. There are three versions at the moment
    - a) version='1' -> Marginal shapley value
    - b) version='2' -> Marginal shapley value (Optimised versions, i.e all the counts of unique rows of dataset are pre calculated)
    - c) version='3' -> Conditional shapley value
    - d) version='4' -> Causal shapley value

## Pre-requisites:
- Run synthetic_data_gen.py
  - uncomment gen_desc() to generate discrete dataset. Modify _add_features() method to add causality in the dataset.
  - for continuous data use gen_dataset()
  - In both cases, supply file name. It creates a csv file under output/dataset/file_name.csv
- Train the model
  - Run train(model_type='classification',file_name='synthetic_discrete_2', save_model=True) by specifying relevant arguments. It creates a folder of output/dataset/file_name under which train and test files are stored.
  - Manually create a json file under output/dataset/file_name with name causal_struct.json with syntax 
    - {
    "0": [ ],
    "1": [ ],
    "2": [
      0,
      1
    ],
    "3": [
      0,
      1,
      2
    ]
}
    - With keys being feature_id and value being the parents of that feature_id
  - Model is saved under output/model/file_name.sav