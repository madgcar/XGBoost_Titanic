# importo las librerias


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge #libreria de regularizacion
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from funpymodeling.exploratory import freq_tbl
import seaborn as sns

# your code here

#load the .env file variables
load_dotenv()
connection_string = os.getenv('DATABASE_URL')
#print(connection_string)

df_raw = pd.read_csv(connection_string)

df = df_raw.copy()

#Paso 2:

#Explore y limpie los datos.
# Transformo la data solo de embarque y de sexo que las necesito categoricas ya que pueden ser 
# importantes como explicativas

df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])


df.dropna()

df['Sex_encoded']=df['Sex'].apply(lambda x: 1 if x == 'female' else 0 )

# Divido los datos de entrenamiento y de validacion para iniciar la exploracion 
# y el EDA
# 2.1 Split the dataset so to avoid bias

X = df.drop(['Survived','Cabin', 'Name', 'Ticket', 'Age', 'Embarked'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=70)

# Uno mi X_train y mi y_train en un unico DataFrame
# 2.2 Join the train sets to ease insights

df_train = pd.concat([X_train,y_train], axis=1)

# 2.10 Perform correlation analysis - Pearson or Point Biserial

X_train.corr().style.background_gradient(cmap='Blues') # si la correlacion es mayor a 75% debe llamarnos la atencion

# Preprocessing --------------------------------------------------------
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Metrics --------------------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

#Construya un primer modelo predictivo usando Random Forest. Elija una métrica de evaluación y luego optimice los hiperparámetros de su modelo.

# 3.1 Create your pipeline processing
# https://stackoverflow.com/questions/61641852/
# https://jaketae.github.io/study/sklearn-pipeline/

cat_cols = X_train.select_dtypes(include='category').columns
num_cols = X_train.select_dtypes(include='number').columns

# hacemos dos transformadores, el Onehotencoder transforma las categoricas a numericas 0 y 1

cat_transformer_d = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse=False))])
cat_transformer_nd = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])
num_transformer = Pipeline(steps=[('scaler',  MinMaxScaler())])
preprocessor_d = ColumnTransformer(transformers=[('num',num_transformer, num_cols),('cat',cat_transformer_d, cat_cols)])
preprocessor_nd = ColumnTransformer(transformers=[('num',num_transformer, num_cols),('cat',cat_transformer_nd, cat_cols)])
encode_data_d = Pipeline(steps=[('preprocessor', preprocessor_d)])
encode_data_nd = Pipeline(steps=[('preprocessor', preprocessor_nd)])



rf_reg = Pipeline(steps=[('preprocessor', preprocessor_nd), ('regressor', RandomForestRegressor())])
rf_reg.fit(X_train,y_train)
print(f'R2 score: {rf_reg.score(X_train,y_train)}')

