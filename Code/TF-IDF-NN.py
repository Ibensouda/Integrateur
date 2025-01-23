import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker
import joblib

file_path = 'cleaned_dataset.csv'
dataset = pd.read_csv(file_path)

texts = dataset['MainText']
labels = dataset['label']

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

#stratify=encoded_labels pour répartir équitablement le nombre de phishing sms dans train et test
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

#stop_words=None parce que en sms chaque mot est important l'orthographe de mots récurrents peut être importante 
#j'ai choisi la norme l2 pcq ça marche un peu mieux que l1 
vectoriseur = TfidfVectorizer(max_df=.80, stop_words=None, use_idf=True, norm="l2")

#fit_transform train et transform test
X_train_tfidf = vectoriseur.fit_transform(X_train)
X_test_tfidf = vectoriseur.transform(X_test)

#Ratio de ham et spam sms pour train et test 
train_ratios = pd.Series(y_train).value_counts(normalize=True)*100
test_ratios = pd.Series(y_test).value_counts(normalize=True)*100

label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
train_ratios.index = train_ratios.index.map(label_mapping)
test_ratios.index = test_ratios.index.map(label_mapping)
#équitablement réparti 


print(train_ratios, test_ratios)

#modèle classique de NN je mets des couches denses pcq pas besoin de compliquer à mon avis
model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),  
    Dense(128, activation='relu'),         
    Dropout(0.3),                          
    Dense(64, activation='relu'),           
    Dropout(0.3),                          
    Dense(1, activation='sigmoid')         
])
print(X_train_tfidf.shape[1])

tracker = EmissionsTracker()
tracker.start()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_tfidf, y_train,
    validation_data=(X_test_tfidf, y_test),
    epochs=10, batch_size=32, verbose=1
)

result = model.evaluate(X_test_tfidf, y_test, verbose=0)
tracker.stop()
print("Test loss:", result[0])
print("Test accuracy:", result[1])


y_pred = (model.predict(X_test_tfidf) > 0.5).astype("int32")


cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig("confusion_matrix.png")
plt.show()
