import os
import shutil
import random

#ana dataset klasörü
BASE_DIR = "dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
VAL_DIR = os.path.join(BASE_DIR, "Validation")

os.makedirs(VAL_DIR, exist_ok=True) #validation klasörünü oluşturur

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for cls in classes:
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True) #her sınıf için validation klasör oluşturur
    
VAL_SPLIT = 0.15 
print("validation ayırma işlemleri başladı...")

for cls in classes:
    train_class_path = os.path.join(TRAIN_DIR, cls)
    val_class_path = os.path.join(VAL_DIR,cls)
    
    files = os.listdir(train_class_path) #o sınıftaki tüm dosyaları listeler
    random.shuffle(files)
    
    val_count = int(len(files) * VAL_SPLIT) #validation için ayrılacak dosya sayısı
    val_files = files[:val_count]
    
    print(f"{cls}: {val_count} dosya validation klasörüne taşınıyor...")
    
    for file in val_files:
        src = os.path.join(train_class_path, file)
        dst = os.path.join(val_class_path, file)
        
        shutil.move(src, dst)
        
print("\n işlem tamamlandı")
print("klasör yapısı:")

print("""
      dataset/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Validation/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
        """)
            