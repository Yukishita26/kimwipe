from PIL import Image as img
import numpy as np
import glob

kimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/kimwipe/*")]
likekimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/likekimwipe/*")]
verykimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/verykimwipe/*")]
notkimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/notkimwipe/*")]

features = np.array(kimwipes + notkimwipes)
labels = np.concatenate([np.zeros(len(kimwipes)),np.ones(len(notkimwipes))])
features2 = np.array(verykimwipes + likekimwipes + notkimwipes)
labels2 = np.concatenate([np.ones(len(verykimwipes))*2,np.ones(len(likekimwipes)),np.zeros(len(notkimwipes))])

print(features.shape, labels.shape)
print(features2.shape, labels2.shape)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
import pickle

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
from sklearn.decomposition import NMF

from sklearn.model_selection import KFold
def cross_val_scores(X, y, clf, cv=5):
    kf = KFold(n_splits=cv)
    print("called")
    trains = []
    tests = []
    for train_i, test_i in kf.split(X):
        train_x, train_y, test_x, test_y = X[train_i], y[train_i], X[test_i], y[test_i]
        clf.fit(train_x, train_y)
        trains += [clf.score(train_x, train_y)]
        tests += [clf.score(test_x, test_y)]
        print("i")
    return np.array(trains), np.array(tests)

if 0:
    s_max=0
    s_maxi=-1
    for i in [0]:
        print(i)
        if i==0:
            tranced = features
        else:
            #nmf = NMF(n_components=i, init='random', random_state=0, tol=0, max_iter=2000)
            #tranced = nmf.fit_transform(features)
            pca = PCA(n_components=i)
            tranced = pca.fit_transform(features2)
        print("PCA has completed")
        #train_x, test_x, train_y, test_y = train_test_split(tranced, labels, test_size=0.2, random_state=0)
        clf = ExtendedForestClassifier(random_state=0, boosting=False, bootstrap=True, bootstrap_features=True, n_estimators=10)
        #clf.fit(train_x, train_y)
        trains, tests = cross_val_scores(features2, labels2, clf)
        print("train", trains.mean())
        print("test", tests.mean())
        if tests.mean()>s_max:
            s_max = tests.mean()
            s_maxi = i
    print(s_maxi, s_max)


clf = ExtendedForestClassifier(random_state=0, 
                                boosting=False, bootstrap=True, bootstrap_features=True, n_estimators=100)
clf.fit(features2, labels2)
with open('classifier.bin', mode='wb') as f:
    pickle.dump(clf, f)
print(clf.score(features2, labels2))

