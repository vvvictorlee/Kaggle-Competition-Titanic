# Kaggle-Competition-Titanic

This is a project in an IPython Notebook for the Kaggle competition, Titanic Machine Learning From Disaster. The goal of this repository is to implement classic machine learning algorithms from scratch and also to provide a comparative analysis using SciKit-Learn for the people who are interested in machine learning.

The best score is achieved by SVC algorithm placed 987th out of all 6912 submissions, top 15%. 

###   Installation:

To run this notebook interactively, you need to download the following dependencies:
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)

### Usage

After downloading the files above, you can go to the project directory and open ipython in your terminal using 'jupyter notebook'. You can enter the following notebooks to see how the project was done and execute each cell by `Shift + return`.

1. `Data_Preprocessing.ipynb`: how titanic training and test data are transformed into categorical and one-hot encoded data
2. `Implemented_Model_Performance.ipynb`: all implemented models are applied in ths notebook
3. `Sklearn_Models_Performance.ipynb`: all Sklearn models are used and tuned in this notebook

**Data**

1. `train.csv`: original training data.
2. `train_processed.csv`: transformed training data
3. `test.csv`: original test data.
4. `test_processed.csv`: transformed test data
5. `gender_submission.csv`: the format to submit predictions to Kaggle Leaderboard, the Survival labels in this file are not real, they are just randomly filled in to show the format. 
6. `titanic.csv`: the submission file to Kaggle Leaderboard

**Dependent models: our implementations**

This directory has 7 models:

1. `decisionTree_GR.py`
2. `decisionTree_IG.py`
3. `kmeans.py`
4. `kmodes.py`
5. `multi_perceptron.py`
6. `NaiveBayers.py`
7. `perceptron.py`

Because KNN is simple, we implemented KNN inside the `Sklearn_Models_Performance.ipynb`.

**Project Report**

- `FinalReport378.pdf`

###   Kaggle Competition | Titanic Machine Learning from Disaster

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Because there were not enough lifeboats for the passengers and crew, some groups of people might be more likely to survive than the others. In this project, we want to investigate the factors that lead to the survival of people. We would apply several tools of data mining algorithms to predict which passengers survived the tragedy. The result of our predictions can become a good source to improve the survival rate of passengers and crew.

####  The components of the project contains the following steps:
####  Data Handling
*   Importing Data with Pandas
*   Extracting new features
*   Handling missing values 
*   Transforming attributes into categorical type
*   Encode features using One hot encoding
*   Feature Selection

####  Machine Learning Algorithms Implementations from scratch
*   Implementing algorithms including 
      - Naive Bayers
      - Decision Tree
      - Perceptron
      - KNN
      - K-means and K-modes with KNN 
*   Compare results using accuracy

####  Data Analysis using SciKit-Learn
*    Supervised Machine learning Techniques:
      - Random Forest 
      - Decison Tree
      - Support Vector Machine (SVM)
      - KNN
      - Naive Bayers (both Gaussian and Multinominal)
      - Perceptron
      - SGD
      - Logistic Regression
      - MLP

####  Valuation of the Analysis
*   10-folds cross validation to evaluate results locally
*   Output the results from the IPython Notebook to Kaggle

#### Scores 

Our Implementations on Kaggle Leaderboard

All Sklearn Models on Training Dataset

Top 5 Sklearn Models on Kaggle Leaderboard

Competition Website: http://www.kaggle.com/c/titanic-gettingStarted
