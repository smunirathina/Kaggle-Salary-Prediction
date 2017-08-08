
#Gradient Boosting Tree
import os, sys, glob
import pandas as pd
import numpy as np
import _pickle as pickle
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def loadTrain():
    """jobId, companyId, jobType, degree, major, industry, yearsExperience,
       milesFromMetropolis, salary
    """
    trainFeatures = pd.read_csv('C:\\smunirathina\\Profile Backup1\\Sathyan\\Indeed\\indeed_data_science_exercise\\backup\\train_features_2013-03-07.csv')
    trainSalaries = pd.read_csv('C:\\smunirathina\\Profile Backup1\\Sathyan\\Indeed\\indeed_data_science_exercise\\backup\\train_salaries_2013-03-07.csv')
    data = pd.merge(trainFeatures, trainSalaries, on='jobId')
    Y = data['salary'].values
    extractors = {'companyId': CountVectorizer(),
                     'jobType': CountVectorizer(),
                     'degree': CountVectorizer(),
                     'major': CountVectorizer(),
                     'industry': CountVectorizer()}
    X = []
    column = []
    for columnName, extractor in extractors.items():
        x1 = extractor.fit_transform(data[columnName])
        X.append(x1.toarray())
        for v in sorted(extractor.vocabulary_, key=extractor.vocabulary_.get):
            column.append(columnName[0]+'_'+v)
    X.append(data[['yearsExperience','milesFromMetropolis']].values)
    column.append('yearsExperience')
    column.append('milesFromMetropolis')
    scaler = StandardScaler()
    X = scaler.fit_transform(np.concatenate(X, axis=1))
    return X, Y, extractors, scaler, column

def loadTest(extractors, scaler):
    data = pd.read_csv('C:\\smunirathina\\Profile Backup1\\Sathyan\\Indeed\\indeed_data_science_exercise\\backup\\test_features_2013-03-07.csv')
    X = []
    for columnName, extractor in extractors.items():
        x1 = extractor.transform(data[columnName])
        X.append(x1.toarray())
    X.append(data[['yearsExperience','milesFromMetropolis']].values)
    X = scaler.transform(np.concatenate(X, axis=1))
    return X, data['jobId']


def heldoutScore(clf, X_test, y_test):
    mse = []
    for y_pred in clf.staged_decision_function(X_test):
        mse.append(mean_squared_error(y_test, y_pred))
    mse = np.array(mse)
    idx=mse.argmin()
    return mse[idx], idx

def tuneGBMparam(X,Y):
    paramGrid = {'max_depth': [5,6,7],
            'min_samples_leaf': [15,20,25]}
    est = GradientBoostingRegressor(n_estimators=100, loss='ls', learning_rate=0.1)
    gs = GridSearchCV(est, paramGrid, scoring='mean_squared_error', n_jobs=9).fit(X, Y)  #n_jobs=9
    return gs

def runGBM():
    X, Y, extractors, scaler, column = loadTrain()
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, train_size=0.1, test_size=0.025, random_state=123)
    if os.path.exists('params2.pkl'):
        with open('params2.pkl', 'rb') as f:
            gs = pickle.load(f)
    else:
        gs = tuneGBMparam(xTrain,yTrain)
        with open('params2.pkl', 'wb') as f:
            pickle.dump(gs, f, -1)
    print(gs.best_params_)
    print(gs.best_score_)
    print(gs.grid_scores_)
    md = gs.best_params_['max_depth']
    ms = gs.best_params_['min_samples_leaf']

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=123)
    clf = GradientBoostingRegressor(n_estimators=500, loss='ls', learning_rate=0.05, subsample=0.7,
            max_depth=md, min_samples_leaf=ms, verbose=0)
    clf.fit(xTrain,yTrain)
    pklName = 'gbm_t0.8_md%d_ms%d.pkl' % (md, ms)
    _ = joblib.dump(clf, pklName, compress=1)
    yTrainPred = clf.predict(xTrain)
    yTestPred = clf.predict(xTest)
    mseTrain = mean_squared_error(yTrainPred, yTrain)
    mseTest = mean_squared_error(yTestPred, yTest)
    maeTrain = mean_absolute_error(yTrainPred, yTrain)
    maeTest = mean_absolute_error(yTestPred, yTest)
    print('MSE:', mseTrain, mseTest)
    print('MAE:', maeTrain, maeTest)

    mseTest, idx = heldoutScore(clf, xTest, yTest)
    print(mseTest, idx)

    xTest, id = loadTest(extractors, scaler)
    for i, yPred in enumerate(clf.staged_decision_function(xTest)):
        if i == idx:
            break

    df = pd.DataFrame({'jobId': id, 'salary': yPred[:,0]})
    df.to_csv('C:\\smunirathina\\Profile Backup1\\Sathyan\\Indeed\\indeed_data_science_exercise\\backup\\test_salaries.csv', index=False)

if __name__=="__main__":
    runGBM()
