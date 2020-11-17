import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import featureextraction
from pickle import dump, load
from sklearn.utils import class_weight

#with open("static/enpred_Scale.pkl", 'rb') as file:
#    std_scale = pickle.load(file)
dataset = pd.read_csv('./static/Features for Webserver.csv')
#dataset = pd.read_csv('./Data sets/Types of  promoters/types.csv')
#dataset = pd.read_csv('./Data sets/Strong vs  Weak/feature extraction/Strong Vs Weak.csv')

X = dataset['Sequence'].values
y1 = pd.get_dummies(dataset['Promoter']).values
y2 = pd.get_dummies(dataset['Sigma']).values
y3 = pd.get_dummies(dataset['Strength']).values

y2 = y2[ np.argmax(y1,axis=1)==1]
y3 = y3[ np.argmax(y1,axis=1)==1]
#X_features = np.load('X_features.npy')

fe = featureextraction.extractFeatures()

#np.save(X_features, './static/X_features.npy'))
if True:
    n = 0
    X_features = []
    for seq in X:
        seq = seq.strip(' ')
        X_features.append(fe.calcFV(seq.lower()))
        #try:
            #if X_features[-1].shape[0] == 525:
                #pass
        #except:
            #d = (featureextraction.extractFeatures.feature_result(X[s].strip(' ')))
            #X_features[s] = d
            #print(d)
            #print(n)
        n += 1
    np.save('./static/X_features.npy',X_features)
X_features = np.array(np.load('./static/X_features.npy',allow_pickle=True))

X_features2 = []
for f in X_features:
    if f.shape[0] == 525:
        X_features2.append( f )
    else:
        print(f.shape)
# Standard Scaler
scaler = StandardScaler()
scaler.fit(X_features2)
X_std = scaler.transform(X_features2)
dump(scaler,'./static/data_scale.pkl')

X2_std = X_std[np.argmax(y1,axis=1)==1]
print(X_std.shape)
print(X2_std.shape)

#clf = MLPClassifier(hidden_layer_sizes=(100,20),random_state=1,max_iter=1000,early_stopping=True,validation_fraction=0.3)
#clf = MLPClassifier(hidden_layer_sizes=(100,20),random_state=1,max_iter=1000,early_stopping=True)
#clf = MLPClassifier(random_state=1,max_iter=1000,early_stopping=True)
clf1 = DecisionTreeClassifier(min_samples_split=3,class_weight="balanced")
#clf = clf.fit(X_train,y_train)
clf1.fit(X_std,y1)

X_train,X_test,y_train,y_test = train_test_split(X_std,y1,test_size=0.3)

print(clf1.score(X_train,y_train))
print(clf1.score(X_test,y_test))

dump(clf1,'./static/model_p_np_tree.pkl')

clf2 = DecisionTreeClassifier(min_samples_split=3,class_weight="balanced")
clf2.fit(X2_std,y2)

X_train,X_test,y_train,y_test = train_test_split(X2_std,y2,test_size=0.3)
print(clf2.score(X_train,y_train))
print(clf2.score(X_test,y_test))

dump(clf2,'./static/model_sigma_tree.pkl')

clf3 = DecisionTreeClassifier(min_samples_split=3,class_weight="balanced")
clf3.fit(X2_std,y3)

X_train,X_test,y_train,y_test = train_test_split(X2_std,y3,test_size=0.3)

print(clf3.score(X_train,y_train))
print(clf3.score(X_test,y_test))

dump(clf3,'./static/model_strength_tree.pkl')
