!pip install pymongo
!pip install ucimlrepo
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import oracledb
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets

# 2. Vizualizarea inițială a datelor
df = pd.DataFrame(X)
print("Date inițiale:")
print(df)
# Afișarea coloanelor numerice
numerical_columns = df.select_dtypes(include=['number'])
print("\nColoane numerice:")
print(numerical_columns.columns)

# Afișarea coloanelor non-numerice
non_numerical_columns = df.select_dtypes(exclude=['number'])
print("\nColoane non-numerice:")
print(non_numerical_columns.columns)
# Afișarea numărului de valori lipsă pe coloană
missing_values = df.isnull().sum()
print("Valori lipsă pe fiecare coloană:")
print(missing_values)
# Identificarea rândurilor duplicate
duplicates = df[df.duplicated()]
print("\nRânduri duplicate:")
print(duplicates)
# Selectarea variabilelor categoriale
categorical_columns = df.select_dtypes(include=['object', 'category'])
print("\nVariabile categoriale:")
print(categorical_columns.columns)

# Afișarea valorilor din variabilele categoriale
for col in categorical_columns.columns:
    print(f"\nColoana '{col}':")
    print(df[col].unique())

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Filtrarea datelor pentru categoria "services"
df_services = df[(df['Mjob'] == 'services') | (df['Fjob'] == 'services')]

# 2. Crearea unui bar plot pentru numărul de persoane care fac fiecare tip de serviciu
plt.figure(figsize=(8, 6))
sns.countplot(data=df_services, x='Mjob', order=df_services['Mjob'].value_counts().index)

# 3. Setarea titlului și a etichetelor
plt.title('Numărul de persoane care lucrează în fiecare tip de serviciu')
plt.xlabel('Tipul de serviciu')
plt.ylabel('Numărul de persoane')

# 4. Afișarea plot-ului
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Crearea unei histograme pentru fiecare coloană numerică
numerical_columns = df.select_dtypes(include=['number'])

# Plot pentru fiecare coloană numerică
for col in numerical_columns.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)  # kde=True adaugă o linie de densitate
    plt.title(f'Histograma pentru {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Crearea unui boxplot pentru distribuția vârstei în funcție de activități
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='absences', y='age')


# 2. Setarea titlului și a etic

from sklearn.preprocessing import StandardScaler

# Crearea unui obiect StandardScaler
scaler = StandardScaler()

# Aplicarea standardizării pe datele numerice
df[numerical_columns.columns] = scaler.fit_transform(df[numerical_columns.columns])

# Vizualizarea datelor scalate
print("\nDatele scalate (StandardScaler):")
print(df.head())

# Normalizarea

# Tratamentul valorilor lipsă (înlocuire cu media)
df['Medu'].fillna(df['Medu'].mean(), inplace=True)

# Codificarea variabilelor categorice (Label Encoding)
encoder = LabelEncoder()
df['school'] = encoder.fit_transform(df['school'])  # GP -> 0, MS -> 1
df['sex'] = encoder.fit_transform(df['sex'])        # F -> 0, M -> 1
df['address'] = encoder.fit_transform(df['address'])  # U -> 1, R -> 0

# Normalizarea datelor numerice (scalare între 0 și 1)
scaler = MinMaxScaler()
df[['age', 'Medu', 'absences']] = scaler.fit_transform(df[['age', 'Medu', 'absences']])

# Vizualizarea datelor procesate
print("\nDate procesate:")
print(df)

# Calcularea statisticilor descriptive pentru datele numerice
dfd = pd.DataFrame(X)

statistici_descriptive = dfd.describe()

print("\nStatistici descriptive:")
print(statistici_descriptive)


df = pd.DataFrame(X)

print(df.dtypes)

import pandas as pd

# Presupunem că ai deja un DataFrame numit df
for column in df.columns:
    print(f"Coloana: {column}")
    print(df[column].unique())  # Afișează toate valorile unice din fiecare coloană
    print("-" * 50)  # Separator pentru o mai bună lizibilitate

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(X)


encoder = LabelEncoder()

# Lista coloanelor categorice de tip string (după cum reiese din datele tale)
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                       'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                       'internet', 'romantic']

# Codificăm fiecare coloană categorică
for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

# 2. Calcularea corelației Pearson între variabilele numerice
corelatie = df.corr()

# 3. Afișarea corelației
print("\nCorelația dintre variabilele numerice:")
print(corelatie)

# 3. Calcularea asimetriei (Skewness) și aplatizării (Kurtosis)
asimetria = df.skew()
aplatisirea = df.kurt()

print("\nAsimetria datelor:")
print(asimetria)

print("\nAplatizarea datelor:")
print(aplatisirea)

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Lista variabilelor categorice
categorical_columns = ['school', 'sex', 'famsize', 'Mjob', 'Fjob',
                       'activities', 'higher', 'internet', 'romantic']

# 1. Crearea unei copii a DataFrame-ului pentru a evita modificarea directă
df_categorical = df[categorical_columns].copy()

# 2. Aplicarea Label Encoding pentru a transforma variabilele categorice în numerice
encoder = LabelEncoder()
for column in categorical_columns:
    df_categorical[column] = encoder.fit_transform(df_categorical[column])

# 3. Calcularea corelației Pearson între variabilele numerice
correlation_matrix = df_categorical.corr()

# 4. Crearea heatmap-ului pentru corelațiile dintre variabilele categorice
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# 5. Setarea titlului și etichetelor
plt.title('Corelația între variabilele categorice')
plt.xlabel('Variabile categorice')
plt.ylabel('Variabile categorice')

# 6. Afișarea heatmap-ului
plt.show()
