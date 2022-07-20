# importo las librerias

import xgboost as xgb
from xgboost import XGBClassifier

X = df_train.drop(['Survived','Sex_encoded'], axis=1)
y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=70)

# Importo las librerias

import xgboost as xgb
from xgboost import XGBClassifier

# Creo el modelo

model3 = XGBClassifier()

model3.fit(X,y)

import matplotlib.pyplot as plt
xgb.plot_importance(model3, ax= plt.gca())

predict = model3.predict(X_test)

model3.score(X_train, y_train)

pred_train = model3.predict(X_train)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, pred_train)