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
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
import pickle


from sklearn.ensemble import RandomForestClassifier

for i in [0,5,10,100,842]:
    """
    parameters = {
        "random_state":[0],
        "solver":["adam"],
        "alpha": [0.0001, 0.001, 0.01], 
        "hidden_layer_sizes": [(10,), (100,), (1000,), (10,10), (100,100), (100,100,100)]
    }
    grd = GridSearchCV(MLPClassifier(), parameters, cv=5)
    """
    if i==0:
        tranced = features        
    else:
        pca = PCA(n_components=i)
        tranced = pca.fit_transform(features)
    train_x, test_x, train_y, test_y = train_test_split(tranced, labels, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(train_x, train_y)
    print(i)
    print("train", clf.score(train_x, train_y))
    print("test", clf.score(test_x, test_y))

    



