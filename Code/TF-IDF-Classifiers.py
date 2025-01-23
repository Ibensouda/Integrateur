import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


dataset = pd.read_csv('../Datasets/Fichier_000.csv')

texts = dataset['MainText']
labels = dataset['label']

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Diviser le dataset en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# Extraction des caractéristiques TF-IDF
vectoriseur = TfidfVectorizer(max_df=0.80, stop_words=None, use_idf=True, norm="l2")
X_train_tfidf = vectoriseur.fit_transform(X_train)
X_test_tfidf = vectoriseur.transform(X_test)

# Dossier de sauvegarde des modèles
save_path = '../models/'

# Sauvegarder le vectoriseur
joblib.dump(vectoriseur, f"{save_path}/vectorizer.joblib")
print(f"Vectorizer sauvegardé dans {save_path}/vectorizer.joblib")

# Sauvegarder le label_encoder
joblib.dump(label_encoder, f"{save_path}/label_encoder.joblib")
print(f"Label Encoder sauvegardé dans {save_path}/label_encoder.joblib")

# Fonction pour effectuer une recherche de grille, évaluer et sauvegarder un modèle
def grid_search_evaluate_and_save(model, param_grid, model_name, save_path):
    print(f"\nOptimisation et évaluation du modèle: {model_name}")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train)
    
    print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Affichage des métriques
    print(f"Exactitude: {accuracy:.4f}")
    print(f"Précision: {precision:.4f}")
    print(f"Rappel: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Rapport de classification:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Matrice de confusion pour {model_name}:\n{conf_matrix}")
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot()
    
    # Sauvegarder le modèle optimisé
    joblib.dump(best_model, f"{save_path}/{model_name}_best_model.joblib")
    print(f"Modèle {model_name} sauvegardé dans {save_path}/{model_name}_best_model.joblib")

    # Sauvegarde de la matrice de confusion
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    conf_matrix_df.to_csv(f"{save_path}/{model_name}_confusion_matrix.csv")
    print(f"Matrice de confusion sauvegardée dans {save_path}/{model_name}_confusion_matrix.csv")

    return accuracy, precision, recall, f1, grid_search.best_params_

# Suivi des émissions de CO2
tracker = EmissionsTracker()
tracker.start()

# Définir les grilles de paramètres
param_grids = {
    'Random Forest': {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [5],
        'min_samples_leaf': [1]
    }
}

# Initialisation des modèles
models = {
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Optimisation, évaluation et sauvegarde des modèles
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1, best_params = grid_search_evaluate_and_save(model, param_grids[model_name], model_name, save_path)
    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_params": best_params
    }

# Suivi des émissions de CO2
tracker.stop()

# Résumé des performances
#Ecriture dans le fichier performances.txt
with open('performances.txt', 'w') as file:
    file.write("\nRésumé des performances:\n")
    
    for model_name, result in results.items():
        file.write(f"{model_name}:\n")
        file.write(f"  Exactitude: {result['accuracy']:.4f}\n")
        file.write(f"  Précision: {result['precision']:.4f}\n")
        file.write(f"  Rappel: {result['recall']:.4f}\n")
        file.write(f"  F1-Score: {result['f1']:.4f}\n")
        file.write(f"  Meilleurs paramètres: {result['best_params']}\n")

#Ecriture dans la console 
print("\nRésumé des performances:")
for model_name, result in results.items():
    print(f"{model_name}:\n  Exactitude: {result['accuracy']:.4f}\n  Précision: {result['precision']:.4f}\n  Rappel: {result['recall']:.4f}\n  F1-Score: {result['f1']:.4f}\n  Meilleurs paramètres: {result['best_params']}")

# Analyse des fréquences des mots
print("\nAnalyse des fréquences des mots :")
words = vectoriseur.get_feature_names_out()
word_frequencies = np.asarray(X_train_tfidf.sum(axis=0)).flatten()
word_freq_df = pd.DataFrame({
    'word': words,
    'frequency': word_frequencies
}).sort_values(by='frequency', ascending=False)

# Exporter le DataFrame dans un fichier CSV
word_freq_csv_path = f"{save_path}/word_frequencies.csv"
word_freq_df.to_csv(word_freq_csv_path, index=False)
print(f"Les fréquences des mots ont été sauvegardées dans {word_freq_csv_path}")

# Afficher les 25 mots les plus fréquents
print("\n25 mots les plus fréquents :")
print(word_freq_df.head(25))

# Afficher les 25 mots les moins fréquents
print("\n25 mots les moins fréquents :")
print(word_freq_df.tail(25))

# Analyse spécifique aux messages de type "smishing"
print("\nAnalyse des fréquences des mots pour les messages 'smishing' :")
smishing_texts = dataset[dataset['label'] == 'phishing']['MainText']
smishing_tfidf = vectoriseur.transform(smishing_texts)
smishing_word_frequencies = np.asarray(smishing_tfidf.sum(axis=0)).flatten()
smishing_word_freq_df = pd.DataFrame({
    'word': words,
    'frequency': smishing_word_frequencies
}).sort_values(by='frequency', ascending=False)

# Exporter les fréquences des mots "smishing" dans un fichier CSV
smishing_freq_csv_path = f"{save_path}/smishing_word_frequencies.csv"
smishing_word_freq_df.to_csv(smishing_freq_csv_path, index=False)
print(f"Les fréquences des mots pour les messages 'smishing'")
print(smishing_word_freq_df.head(25))

