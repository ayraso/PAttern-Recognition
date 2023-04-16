
# Kütüphaneler
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2hsv
from skimage.feature import multiscale_basic_features
from sklearn.svm import SVC
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri seti dizini
os.chdir('D:\\4_2-Bahar-22_23\\Pattern Recognition\\Homework03\\HW3\\HWData\\train')

# Etiketleri ve görüntüleri içeren listeleri oluşturma
labels = []
images = []

for label in os.listdir('.'):
    if os.path.isdir(label):
        for image_file in os.listdir(label):
            image_path = os.path.join(label, image_file)
            labels.append(label)
            images.append(image_path)

# Görüntüleri okuma, HSV renk uzayına dönüştürme ve özellikleri çıkarma
features = []

print("> Feature'lar çıkartılıyor.")
for image_file in images:

    # Görüntüyü okuyuma ve HSV renk uzayına dönüştürme
    image = imread(image_file)
    image = resize(image, (224, 224, 3))
    hsv_image = rgb2hsv(image)

    # Özellikleri çıkarma
    feature = multiscale_basic_features(hsv_image, channel_axis=2, sigma_min = 1.00, sigma_max = 16.00,num_sigma = 5)
    features.append(feature)
print("> Feature'lar çıkartıldı.")
# features değişkenindeki özellikleri ndarray yapma
features = np.array(features)
features = features.reshape(features.shape[0], -1)
print("> Model Eğitilmeye Başlandı.")
# SVM modelini eğitme
model = SVC()
model.fit(features, labels)
print("> Model Eğitildi.")

# Test veri seti dizini
test_dir = 'D:\\4_2-Bahar-22_23\\Pattern Recognition\\Homework03\\HW3\\HWData\\test'

# Etiketleri ve test fotolarını içeren listeleri oluşturma
test_labels = []
test_images = []

for label in os.listdir(test_dir):
    if os.path.isdir(os.path.join(test_dir, label)):
        for image_file in os.listdir(os.path.join(test_dir, label)):
            image_path = os.path.join(test_dir, label, image_file)
            test_labels.append(label)
            test_images.append(image_path)

# Test fotolarını okuma, HSV renk uzayına dönüştürme ve özellikleri çıkarma
test_features = []
print("> Feature'lar çıkartılıyor.")
for image_file in test_images:

    # Görüntüyü okuma ve HSV renk uzayına dönüştürme
    image = imread(image_file)
    image = resize(image, (224, 224, 3), anti_aliasing=True)
    hsv_image = rgb2hsv(image)

    # Özellikleri çıkarma
    feature = multiscale_basic_features(hsv_image, channel_axis=2, sigma_min = 1.00, sigma_max = 16.00,num_sigma = 5)
    test_features.append(feature)
print("> Feature'lar çıkartıldı.")
# Özelliklerin boyutlarını ndarray'e çevirme
test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], -1)
print("> Test ediliyor.")
# Modeli kullanarak test verilerini tahmin etme
predictions = model.predict(test_features)
print("> Test edildi.")
# Confusion matrix oluşturun
cm = confusion_matrix(test_labels, predictions)

# Confusion matrixi ısı haritası olarak görselleştirin
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
