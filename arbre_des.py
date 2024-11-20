import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('Carbon_(CO2)_Emissions_by_Country.csv')

# Préparer les données
# Définir la cible : Si les émissions par habitant dépassent 1 tonne métrique
data['High_Emissions'] = (data['Metric Tons Per Capita'] > 1).astype(int)

# Sélectionner les prédicteurs
# Encodage de 'Region'
data_encoded = pd.get_dummies(data[['Region']], drop_first=True)
X = pd.concat([data_encoded, data[['Kilotons of Co2']]], axis=1)
y = data['High_Emissions']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Visualiser l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Low', 'High'],
    filled=True
)
plt.title("Arbre de Décision - Émissions de CO₂", fontsize=16)
plt.savefig('co2_decision_tree.png')  # Sauvegarde de l'image
plt.show()