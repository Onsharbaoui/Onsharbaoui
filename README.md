# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Load a dataset into a Pandas Dataframe
data= pd.read_csv('train.csv')
print("Full train dataset shape is {}".format(data.shape))

data_2=pd.read_csv('test.csv')

data

data.head(5)

data.describe()

data.info()


plot_df = data.Transported.value_counts()
plot_df.plot(kind="bar")

fig, ax = plt.subplots(5,1,  figsize=(10, 10))
plt.subplots_adjust(top = 2)

sns.histplot(data['Age'], color='b', bins=50, ax=ax[0]);
sns.histplot(data['FoodCourt'], color='b', bins=50, ax=ax[1]);
sns.histplot(data['ShoppingMall'], color='b', bins=50, ax=ax[2]);
sns.histplot(data['Spa'], color='b', bins=50, ax=ax[3]);
sns.histplot(data['VRDeck'], color='b', bins=50, ax=ax[4]);

# Prétraitement des données
## Nettoyage des données
# Sélection des colonnes numériques
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Sélection des colonnes non numériques
non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns

# Imputation pour les colonnes numériques
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Imputation pour les colonnes non numériques (par exemple, avec la stratégie 'most_frequent')
imputer = SimpleImputer(strategy='most_frequent')
data[non_numeric_columns] = imputer.fit_transform(data[non_numeric_columns])

df = pd.get_dummies(data, columns=['HomePlanet','Cabin','Destination','Name'])

# Prétraitement des données test
## Nettoyage des données
# Sélection des colonnes numériques existantes dans data_2
numeric_columns_test = data_2.select_dtypes(include=['float64', 'int64']).columns

# Imputation pour les colonnes numériques
imputer_test = SimpleImputer(strategy='mean')
data_2[numeric_columns_test] = imputer_test.fit_transform(data_2[numeric_columns_test])

# Sélection des colonnes non numériques existantes dans data_2
non_numeric_columns_test = data_2.select_dtypes(exclude=['float64', 'int64']).columns

# Imputation pour les colonnes non numériques (par exemple, avec la stratégie 'most_frequent')
imputer_test = SimpleImputer(strategy='most_frequent')
data_2[non_numeric_columns_test] = imputer_test.fit_transform(data_2[non_numeric_columns_test])

df_2 = pd.get_dummies(data_2, columns=['HomePlanet','Cabin','Destination','Name'])

# Sélection des fonctionnalités (toutes les colonnes sauf la dernière qui est la cible)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Maintenant, assurez-vous que X_train et y_train correspondent
print(len(X_train), len(y_train))  # Assurez-vous que ces deux nombres sont identiques

# Application des algorithmes de machine learning
## Choix des algorithmes
tree = DecisionTreeClassifier()
nn = MLPClassifier()


## Entraînement des modèles
nn.fit(X_train, y_train)

## Entraînement des modèles
tree.fit(X_train, y_train)

# Évaluation des modèles
## Métriques d'évaluation
tree_pred = tree.predict(X_test)
nn_pred = nn.predict(X_test)
print("Précision de l'arbre de décision : ", accuracy_score(y_test, tree_pred))
print("Précision du réseau de neurones : ", accuracy_score(y_test, nn_pred))

# Créez un DataFrame avec les prédictions des deux modèles
predictions_df = pd.DataFrame({'Tree_Predictions': tree_pred, 'NN_Predictions': nn_pred})

# Concaténez le DataFrame des prédictions avec l'ensemble de test
X_test_with_predictions = pd.DataFrame(X_test)  # Convertissez X_test en DataFrame si ce n'est pas déjà le cas
X_test_with_predictions = pd.concat([X_test_with_predictions, predictions_df], axis=1)


## Validation croisée
tree_scores = cross_val_score(tree, X_train, y_train, cv=5)
nn_scores = cross_val_score(nn, X_train, y_train, cv=5)

print("Score de validation croisée de l'arbre de décision : ", tree_scores.mean())
print("Score de validation croisée du réseau de neurones : ", nn_scores.mean())

# Prédiction sur les données de test
final_model = tree if tree_scores.mean() > nn_scores.mean() else nn
final_predictions = final_model.predict(X_test)


