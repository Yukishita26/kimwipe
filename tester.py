from PIL import Image as img
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import random
from math import ceil
from sklearn.metrics import r2_score
def mean_with_weight(arr, weight=None, axis=0):# m x n 行列のみ,axis方向に縮退
    h,w = arr.shape # 縦，横
    if 0:
        if weight == None:
            if axis == 0:
                weight = np.ones(h)/h
            else:
                weight = np.ones(w)/w
    whole_weight = sum(weight)
    if len(weight)!=arr.shape[axis]: return "Error!"
    if axis == 0:
        r = np.zeros(w)
        for i in range(w):
            r[i] = sum(arr[j][i]*weight[j] for j in range(h))/whole_weight
    else:
        r = np.zeros(h)
        for i in range(h):
            r[i] = sum(arr[j][i]*weight[j] for j in range(w))/whole_weight
    return r
class ExtendedForestRegressor(BaseEstimator, RegressorMixin):
    #import random
    def __init__(self,
                 b_estimator='d',# decision or extra
                 boosting=False,#if True, AdaBoost, o.w. Bagging
                 random_state=None,
                 bootstrap=True,
                 bootstrap_features=False,
                 max_samples=1.0,
                 n_estimators=100,
                 ):
        if b_estimator=='d' or b_estimator=='decision':
            self.b_estimator = 'decision'
            self.base_estimator = DecisionTreeRegressor()
        elif b_estimator=='e' or b_estimator=='extra':
            self.b_estimator = 'extra'
            self.base_estimator = ExtraTreeRegressor()
        else:
            self.b_estimator = b_estimator
            self.base_estimator = b_estimator
        self.boosting = boosting
        self.random_state = random_state
        if random_state==None: self.random_state = random.randint(0,1000)
        else: self.random_state = random_state
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        
        self.sample_rate = self.max_samples/ceil(self.max_samples)
        if boosting:
            self.estimator_ = AdaBoostRegressor(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                #bootstrap=self.bootstrap,
                #bootstrap_features=self.bootstrap_features,
                #max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        else:
            self.estimator_ = BaggingRegressor(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        #self.estimators_ = None
        #self.feature_importances_ = None

    def fit(self, X, y):
        if self.max_samples > 1:
            X = pd.concat([X]*ceil(self.max_samples))
            y = pd.concat([y]*ceil(self.max_samples))
        self.estimator_.fit(X, y)
        self.estimators_ = self.estimator_.estimators_
        if self.boosting:
            self.estimator_weights_ = self.estimator_.estimator_weights_
            self.estimator_errors_ = self.estimator_.estimator_errors_
            self.feature_importances_s_ = \
                np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = mean_with_weight(self.feature_importances_s_, weight=self.estimator_weights_)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        else:
            self.feature_importances_s_ = np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = self.feature_importances_s_.mean(axis=0)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        return self
    def predict(self, X):
        return self.estimator_.predict(X)
    def score(self, X, y):
        yhat = self.predict(X)
        return r2_score(y, yhat)
    def figure_importances(self, X):
        feature_labels = X.columns
        indices = np.argsort(self.feature_importances_)
        plt.figure(figsize=(6,6))
        plt.barh(range(len(indices)), self.feature_importances_[indices], xerr=self.feature_importances_std_[indices], color='b', align='center')
        plt.yticks(range(len(indices)), feature_labels[indices])
        plt.show()
class ExtendedForestClassifier(BaseEstimator, ClassifierMixin):
    #import random
    def __init__(self,
                 b_estimator='d',# decision or extra
                 boosting=False,#if True, AdaBoost, o.w. Bagging
                 random_state=None,
                 bootstrap=True,
                 bootstrap_features=False,
                 max_samples=1.0,
                 n_estimators=100,
                 ):
        if b_estimator=='d' or b_estimator=='decision':
            self.b_estimator = 'decision'
            self.base_estimator = DecisionTreeClassifier()
        elif b_estimator=='e' or b_estimator=='extra':
            self.b_estimator = 'extra'
            self.base_estimator = ExtraTreeClassifier()
        else:
            self.b_estimator = b_estimator
            self.base_estimator = b_estimator
        self.boosting = boosting
        self.random_state = random_state
        if random_state==None: self.random_state = random.randint(0,1000)
        else: self.random_state = random_state
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        
        self.sample_rate = self.max_samples/ceil(self.max_samples)
        if boosting:
            self.estimator_ = AdaBoostClassifier(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                #bootstrap=self.bootstrap,
                #bootstrap_features=self.bootstrap_features,
                #max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        else:
            self.estimator_ = BaggingClassifier(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        #self.estimators_ = None
        #self.feature_importances_ = None

    def fit(self, X, y):
        if self.max_samples > 1:
            X = pd.concat([X]*ceil(self.max_samples))
            y = pd.concat([y]*ceil(self.max_samples))
        self.estimator_.fit(X, y)
        self.estimators_ = self.estimator_.estimators_
        if self.boosting:
            self.estimator_weights_ = self.estimator_.estimator_weights_
            self.estimator_errors_ = self.estimator_.estimator_errors_
            self.feature_importances_s_ = \
                np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = mean_with_weight(self.feature_importances_s_, weight=self.estimator_weights_)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        else:
            self.feature_importances_s_ = np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = self.feature_importances_s_.mean(axis=0)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        return self
    def predict(self, X):
        return self.estimator_.predict(X)
    def score(self, X, y):
        #yhat = self.predict(X)
        return self.estimator_.score(X, y) #r2_score(y, yhat)
    def figure_importances(self, X):
        feature_labels = X.columns
        indices = np.argsort(self.feature_importances_)
        plt.figure(figsize=(6,6))
        plt.barh(range(len(indices)), self.feature_importances_[indices], xerr=self.feature_importances_std_[indices], color='b', align='center')
        plt.yticks(range(len(indices)), feature_labels[indices])
        plt.show()

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        plt.legend(loc="best")
        return plt

from sklearn.ensemble import RandomForestClassifier

with open('classifier.bin', mode='rb') as f:
    clf = pickle.load(f)

testdir = glob.glob("./images/tester/*")
testimgs = [np.array(img.open(name)).reshape((-1,)) for name in testdir]
testname = [s[16:] for s in testdir]
import shutil
for i,img in enumerate(testimgs):
    ps = [est.predict([img])[0] for est in clf.estimators_]# + ([clf.predict([img])[0]]*10)
    #print(testname[i], ps, clf.predict([img])[0])
    #plt.imshow(img.reshape(64,64,3))
    #plt.title("{}% Kimwipe!".format(100-int(100*sum(ps)/(len(ps)))))
    #plt.show()
    print("{} is {}% Kimwipe!".format(testname[i], 100-int(100*sum(ps)/(len(ps)))), clf.predict([img])[0])
    if clf.predict([img])[0]==2:
        shutil.copy2(testdir[i], "./kim/"+testname[i])
    elif clf.predict([img])[0]==1:
        shutil.copy2(testdir[i], "./lik/"+testname[i])
    else:
        shutil.copy2(testdir[i], "./not/"+testname[i])