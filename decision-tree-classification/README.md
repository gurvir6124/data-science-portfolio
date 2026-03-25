# Tree-Based Machine Learning Model Comparison

This project compares several tree-based machine learning classifiers on a nonlinear binary classification dataset. The aim is to investigate how different models separate the data and how hyperparameter choices influence their performance.

The dataset is generated using the `make_moons` function from scikit-learn, which produces two interleaving classes with added noise. This provides a useful test case for evaluating the ability of different classifiers to capture nonlinear decision boundaries.

Three models are trained and evaluated:

- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost Classifier  

Hyperparameters for each model are explored through a simple grid search. Models are trained on a training set, hyperparameters are selected using a validation set, and final performance is evaluated on a held-out test set. This ensures that the test data is not used during model selection.

Model performance is assessed using metrics such as accuracy and recall. Decision boundaries are then visualised to show how each classifier partitions the feature space and separates the two classes.

