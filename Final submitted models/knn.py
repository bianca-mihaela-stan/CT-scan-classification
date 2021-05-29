import numpy as np
import matplotlib.image as img
import os
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
import time


class KNNClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def calculeaza_distante(self, test_image, metrica):
        if metrica == 'l2':
            distanta = (np.sum((train_images - test_image) ** 2, axis=1))
            distanta = np.sqrt(distanta)
            return distanta
        elif metrica == 'l1':
            distanta = np.sum(np.abs(train_images - test_image), axis=1)
            return distanta
    
    def clasifica_imagine(self, test_image, nr_vecini=5, metrica='l2'):
        distanta = self.calculeaza_distante(test_image, metrica) # Calculez distanta de la imaginea
                                                            # de test la toate imaginile de train.

        indexes = np.argsort(distanta)                      # Iau indecsii distantelor sortate.

        vecini = self.train_labels[indexes[:nr_vecini]]     # Iau primii nr_vecini vecini.

        clase = np.bincount(vecini)                         # Vad de cate ori apare fiecare label.

        return np.argmax(clase)                             # Aleg label-ul cel mai comun.

start = time.time()
f = open("../ai-unibuc-23-31-2021/train.txt")           
lines = f.readlines()                                   

train_images = []                           
train_labels = []

for line in lines:                                          # Citesc imaginile de train.                             
    line = line.strip().split(",")
    image_name = line[0]
    image = img.imread(os.path.join("../ai-unibuc-23-31-2021/train/", image_name))
    image = image.flatten()                                 # Le transform intr-o lista.                  
    image_label = int(line[1])
    train_images.append(image)
    train_labels.append(image_label)

train_images = np.array(train_images)                       
train_labels = np.array(train_labels)


f = open("../ai-unibuc-23-31-2021/validation.txt")
lines = f.readlines()

validation_images = []
validation_labels = []

for line in lines:                                          # La fel pentru datele de validare.
    line = line.strip().split(",")
    image_name = line[0]
    image = img.imread("../ai-unibuc-23-31-2021/validation/"+ image_name)
    image = image.flatten()
    image_label = int(line[1])
    validation_images.append(image)
    validation_labels.append(image_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)


classifier = KNNClassifier(train_images, train_labels)      # Initializez clasificatorul.

predictions = []
for validation_image in validation_images:                  # Clasific imaginile din validare.
    pred_label = classifier.clasifica_imagine(validation_image)
    predictions.append(pred_label)


pred_labels = np.array(predictions)

corecte = np.sum(pred_labels == validation_labels)    # Vad cate predictii sunt corecte.
total = len(validation_labels)

accuracy = corecte / total

print(f'Acccuracy: {accuracy * 100}%')                 # Afisez acuratetea.

# Fac matricea de confuzie.
cf_m = sklearn.metrics.confusion_matrix(y_true = validation_labels, y_pred = pred_labels, labels = [0, 1, 2]) 

print(cf_m)

# Prezic datele de test.
f = open("../ai-unibuc-23-31-2021/test.txt")
g = open("test.txt", "w")
lines = f.readlines()

test_images = []
test_labels = []

for line in lines:
    line = line.strip()
    image_name = line
    image = img.imread("../ai-unibuc-23-31-2021/test/"+ image_name)
    image = image.flatten()
    test_images.append(image)

    pred_label = classifier.clasifica_imagine(image)
    test_labels = pred_label

    g.write(image_name + "," + str(pred_label) + "\n")

stop=time.time()
print(stop-start)                                   # A durat 819 s.