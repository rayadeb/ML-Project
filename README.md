# ML-Project
To use this code, please examine the file explanations below:
1. Models: Folder containing all 3 models, including the test models. The 3 models explained in the paper are "hybrid_nn.py," "rnn.py," and "FNN.py."
2. Preprocessing: Folder containing preprocessed data, raw data, and a folder to place data for future prediction. Includes
two preprocessing scripts for their respective purposes. "preprocessing_adv_team_P_stats.py" is used to preprocess team data, while "preprocessing_player_stats_R_and_P.py" is used to preprocesses individual player data.
3. Results: The results folder contains "Complete_Rankings.csv", which is the input for "metrics.py."
"metrics.py" outputs "overall_accuracy_metrics.csv" and "seasonal_accuracy_metrics.csv." This is true for each sub-folder which specifies the model whose results are within.
"visualize.py" is used to visualize a given season.
