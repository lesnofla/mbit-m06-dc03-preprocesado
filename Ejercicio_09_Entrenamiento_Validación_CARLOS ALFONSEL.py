#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # 9. ENTRENAMIENTO Y VALIDACIÓN.
# 
# Preparar el conjunto de datos para entrenamiento y test; entrenar un algoritmo cualquiera sobre la muestra de entrenamiento y probar el resultado en la muestra de test. De igual manera haciendo primero validación cruzada (K-fold) sobre la muestra de entrenamiento primero.
# 
# ### IMPORTACIÓN DE LIBRERÍAS

# In[ ]:


# Paquetes de manipulación de datos y creación de gráficos:
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 
import numpy as np
import scipy

# Paquetes de visualización:
import matplotlib.pyplot as plt
import seaborn as sns

# Librería de datasets
from sklearn import datasets
from sklearn.datasets import load_iris

# Utilidades de la librería de sklearn necesarias para técnicas de división de muestras (train/val/test):
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Modelos de aprendizaje automático de la librería sklearn:
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Métricas:
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# Librería de preprocesado:
from sklearn import preprocessing

# Librería de pipeline:
from sklearn.pipeline import make_pipeline

# Librerías para ajuste de parámetros:
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Gestión de warnings:
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)


# ### IMPORTACIÓN Y PREPARACIÓN DATASET (CATEGORIZACIÓN, MISSING VALUES Y OUTLIERS)

# In[ ]:


# Importación de fichero CSV, categorizando algunas variables y cambiando las de valor "yes/no" a int64 (dicotómicas):
df_bank = pd.read_csv('./bank.csv', 
                      dtype={'job':'category',
                             'marital':'category',
                             'education':'category',
                             'contact':'category',
                             'month':'category',
                             'poutcome':'category'
                            },
                      converters = {'default': lambda x: 0 if x == "no" else 1,
                                    'housing': lambda x: 0 if x == "no" else 1,
                                    'loan': lambda x: 0 if x == "no" else 1,
                                    'y': lambda x: 0 if x == "no" else 1}
                      
                     )

# Hacemos una copia del dataframe para optimizar código:
df_test = df_bank.copy()

# Renombramos como TARGET la variable objetivo:
df_test = pd.DataFrame(df_test.rename(columns={'y':'TARGET'}))

# Imputamos los valores ausentes del dataset con uno de los métodos que incluye el paquete pandas (ffill):
df_test.fillna(method = 'ffill', inplace = True)

# Definimos una lista con las variables cuyos outliers queremos filtrar:
vars6sigma = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Filtramos los valores atípicos (outliers) de las variables anteriores mediante el método 6σ:
def remove_outliers_six_sigma(X):
    return X[abs(X - np.mean(X)) < 3 * np.std(X)]

for var in vars6sigma:
    df_test = pd.merge(df_test, pd.DataFrame(remove_outliers_six_sigma(df_test[var].values)), 
                   how = 'left', 
                   left_index = True, 
                   right_index = True).rename(columns={0: var + '_6_sigma'})

# Eliminamos las variables con outliers para quedarnos únicamente con las filtradas:
df_test = df_test.drop(vars6sigma, axis = 1)

# Renombramos las variables filtradas quitándoles el sufijo "_6_sigma":
df_test = pd.DataFrame(df_test.rename(columns={'age_6_sigma':'age',
                                               'balance_6_sigma':'balance',
                                               'duration_6_sigma':'duration',
                                               'campaign_6_sigma':'campaign',
                                               'pdays_6_sigma':'pdays',
                                               'previous_6_sigma':'previous'}))

# Imputamos de nuevo los Missing Values que hayan aparecido con el método ffill de pandas:
df_test.fillna(method = 'ffill', inplace = True)

# Cambiamos el tipo de esas variables (float64) a tipo integer (int64):
for i in vars6sigma:
    df_test[i] = df_test[i].astype(np.int64)

# Reordenamos las columnas de nuestro dataframe para que TARGET sea la última:
df_test = df_test[['age','job','marital','education','default','balance','housing','loan','contact','day','month',
                   'duration','campaign','pdays','previous','poutcome','TARGET']]

# Por último, nos quedamos con las SEIS VARIABLES seleccionadas según el apartado anterior para entrenar el modelo:
df_test = df_test.drop(['age','job','marital','education','contact','day','month','pdays','previous','poutcome'], axis = 1)

# YA TENEMOS NUESTRO DATASET PREPARADO:
df_test.info()


# ## ENTRENAMIENTO CON VALIDACIÓN CRUZADA
# 
# ### a) CONSTRUCCIÓN DE DATASETS DE ENTRENAMIENTO Y VALIDACIÓN

# In[ ]:


X = df_test.iloc[:,range(0,6)]
Y = df_test.iloc[:,6]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


# ### b) ENTRENAMIENTO DATASET MUESTRA "TRAIN" Y VALIDACIÓN CONTRA "TEST"

# In[ ]:


clf = svm.SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
clf.score(X_test, y_test)


# ### c) ENTRENAMIENTO MODELO CON VALIDACIÓN CRUZADA (10 iteraciones, métrica por defecto)

# In[ ]:


##clf = svm.SVC(kernel = 'linear', C = 1)


# In[ ]:


# Cambiando el argumento "gamma" de la función svm.SVC a "auto" se acelera el entrenamiento del modelo (y no cambian de
# manera siginificativa los scores):
clf = svm.SVC(gamma = 'auto')

scores1 = cross_val_score(clf, X, Y, cv = 10)
scores1


# ### d) ENTRENAMIENTO MODELO CON VALIDACIÓN CRUZADA (iterador específico, métrica alternativa)

# In[ ]:


skf = StratifiedKFold(n_splits = 10)

scores2 = cross_val_score(clf, X, Y, cv = skf, scoring = 'recall_weighted')
scores2


# ### e) CROSS-VALIDATE CON DOS MÉTRICAS (SIN/CON EVALUACIÓN DEL SET DE TRAINING)
# 
# La función *cross_validate* nos permite especificar varias métricas de evaluación, y obtener estadísticas sobre los tiempos de aprendizaje y validación, así como métricas de calidad para el conjunto de aprendizaje. Los valores los devuelve en un diccionario.

# In[ ]:


##clf = svm.SVC(kernel = 'linear', C = 1, random_state = 0)


# In[ ]:


# PRIMERO HACEMOS LA VALIDACIÓN CRUZADA CON DOS MÉTRICAS SIN EVALUACIÓN DEL TRAINING SET (return_train_score = false):

scoring = ['accuracy', 'recall_weighted']

scores3 = cross_validate(clf, X, Y, scoring = scoring, cv = 10, return_train_score = False)

print(sorted(scores3.keys()))
print('')
print(scores3['test_recall_weighted'])


# In[ ]:


# A CONTINUACIÓN HACEMOS LA VALIDACIÓN CRUZADA CON EVALUACIÓN DEL TRAINING SET (return_train_score = true):

scores4 = cross_validate(clf, X, Y, scoring = scoring, cv = 10, return_train_score = True)

print(sorted(scores4.keys()))
print('')
print(scores4['test_recall_weighted'])


# ### e) X-VALIDATION CON ITERADOR ESPECÍFICO, DOS MÉTRICAS y EVALUACIÓN DEL SET DE TRAINING

# In[ ]:


scores5 = cross_validate(clf, X, Y, scoring = scoring, cv = skf, return_train_score = True)

print(sorted(scores5.keys()))
print()
print(scores5['test_recall_weighted'])


# ## PROCESADO CON PIPELINES

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = 1))
clf.fit(X_train, y_train)
clf.score(X_test, y_test )


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

skf = StratifiedKFold(n_splits = 10)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = 1))
cross_val_score(clf, X, Y, cv = skf)


# ## COMPARACIÓN ENTRE DISTINTOS MODELOS

# In[ ]:


X = df_test.iloc[:,range(0,6)]
Y = df_test.iloc[:,6]

seed = 5

models = []
models.append(('LR'  , LogisticRegression        ()))
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier      ()))
models.append(('CART', DecisionTreeClassifier    ()))
models.append(('NB'  , GaussianNB                ()))
models.append(('SVM' , SVC                       ()))

results = []
names   = []
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.set_size_inches(15, 12)
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:




