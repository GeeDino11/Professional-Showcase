#Imports
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


#Some text
st.title('Streamlit Demo')

st.write("""
### Exploring Diffrent Classifiers
""")

st.write("""
###### By: Matthew Gerardino

---
""")



#Creating Selectbox
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine'))
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

#Defining our datasets
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
        st.write('Read More About the [Iris Dataset Here](https://archive.ics.uci.edu/ml/datasets/iris)')
        df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
        st.dataframe(df.head())
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
        st.write('Read More About the [Breast Cancer Dataset Here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))')
        df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
        st.dataframe(df.head())
    else: 
        data = datasets.load_wine()
        st.write('Read More About the [Wine Dataset Here](https://archive.ics.uci.edu/ml/datasets/wine)')
        df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
        st.dataframe(df.head())
    
    X= data.data
    y=data.target
    return X,y

st.write(f'## Viewing the {dataset_name} Dataset')

X,y = get_dataset(dataset_name)
st.write('**Shape of Dataset**', X.shape)
st.write('**Number of Classes**', len(np.unique(y)))

#Adding our params
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name =='KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name =='SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        n_estimators = st.sidebar.slider('Num of Estimators', 1, 100)
        params['Max Depth'] = max_depth
        params['Num of Estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name =='SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['Num of Estimators'],
                                    max_depth=params['Max Depth'], random_state=1)

    return clf

clf = get_classifier(classifier_name, params)


# Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = round(accuracy_score(y_test, y_pred), 2)

st.write(f'**Classifier Type: {classifier_name}**')
st.write(f'**Accuracy Score: {acc}**')

# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.title(f'{dataset_name} Data Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# cbar = plt.colorbar()
# cbar.solids.set_edgecolor("face")

#plt.show()
st.pyplot(fig)