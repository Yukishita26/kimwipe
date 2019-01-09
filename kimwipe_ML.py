from PIL import Image as img
import numpy as np
import glob

kimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/kimwipe/*")]
notkimwipes = [np.array(img.open(name)).reshape((-1,)) for name in glob.glob("./images/notkimwipe/*")]

features = np.array(kimwipes+notkimwipes)
labels = np.concatenate([np.zeros(len(kimwipes)),np.ones(len(notkimwipes))])

print(features.shape, labels.shape)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle

grds = []
for i in [0,5,10,100,1000]:
    parameters = {
        "random_state":[0],
        "solver":["adam"],
        "alpha": [0.0001, 0.001, 0.01], 
        "hidden_layer_sizes": [(10,), (100,), (1000,), (10,10), (100,100), (100,100,100)]
    }
    grd = GridSearchCV(MLPClassifier(), parameters, cv=5)
    if i==0:
        tranced = features        
    else:
        pca = PCA(n_components=100)
        tranced = pca.fit_transform(features)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=0)
    grd.fit(tranced, labels)
    grds += [grd]
    predictor = grd.best_estimator_
    predictor.fit(train_x, train_y)
    print(i)
    print("train", predictor.score(train_x, train_y))
    print("test", predictor.score(test_x, test_y))

with open('gridSerches', mode='wb') as f:
    pickle.dump(grds, f)
    


#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(random_state=0, n_estimators=100)


