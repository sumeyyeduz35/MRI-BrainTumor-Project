import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.dataloader import get_loaders
from src.models.cnn_model import BrainTumorCNN


# -------------------------------------------------------
# 1) Eğitim ve doğrulama metriklerini çizdirme fonksiyonu
# -------------------------------------------------------
def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim ve Doğrulama Loss Grafiği")
    plt.legend()
    plt.show()

    # Accuracy grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Eğitim ve Doğrulama Accuracy Grafiği")
    plt.legend()
    plt.show()



# -------------------------------------------------------
# 2) F1-Score + Confusion Matrix (Test Değerlendirme)
# -------------------------------------------------------
def evaluate_model(model, test_loader, classes, device="cpu"):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\nTEST SONUÇLARI — F1-SCORE, PRECISION, RECALL\n")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()



# -------------------------------------------------------
# 3) CNN Eğitimi (early stopping + LR scheduler dahil)
# -------------------------------------------------------
def train_model(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    device="cpu",
    epochs=10
):

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 4  # 4 epoch kötüleşme gelince durdurur

    model.to(device)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # -------------------- TRAIN --------------------
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="Eğitiliyor..."):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)

        # -------------------- VALIDATION --------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        # Console çıktıları
        print(f"Eğitim Loss: {train_loss:.4f} | Eğitim Acc: {train_acc:.2f}%")
        print(f"Doğrulama Loss: {val_loss:.4f} | Doğrulama Acc: {val_acc:.2f}%")

        # Metrikleri kaydet
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # LR Scheduler
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # En iyi modeli kaydet
            torch.save(model.state_dict(), "best_cnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early Stopping tetiklendi!")
                break

    return model, history



# -------------------------------------------------------
# 4) MAIN — Ana Eğitim Çalıştırma
# -------------------------------------------------------
if __name__ == "__main__":

    # -------------------- Dosya Yolları --------------------
    train_dir = "dataset/Training"
    val_dir   = "dataset/Validation"
    test_dir  = "dataset/Testing"

    # -------------------- Dataloader --------------------
    train_loader, val_loader, test_loader, classes = get_loaders(
        train_dir, val_dir, test_dir, batch_size=32
    )

    print("Sınıflar:", classes)

    # -------------------- Model --------------------
    model = BrainTumorCNN(num_classes=len(classes))

    # -------------------- Loss, Optimizer, Scheduler --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2, verbose=True
    )

    # -------------------- GPU / CPU --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    # -------------------- Eğitim --------------------
    model, history = train_model(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        epochs=15
    )

    # -------------------- Eğitim Grafiklerini Çiz --------------------
    plot_metrics(history)

    # -------------------- En iyi modeli yükle --------------------
    model.load_state_dict(torch.load("best_cnn_model.pth"))

    # -------------------- Test Aşaması --------------------
    evaluate_model(model, test_loader, classes, device=device)

    print("\nEğitim ve test tamamlandı. En iyi model: best_cnn_model.pth")
