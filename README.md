
# MRI Beyin Tümörü Sınıflandırma ve Klinik Analiz Raporlama Sistemi
# MRI Brain Tumor Classification and Clinical Analysis Reporting System

Beyin MR görüntülerini derin öğrenme yöntemleriyle analiz ederek tümör kategorilerini sınıflandırmayı ve elde edilen bulguları doktorların değerlendirebileceği yapılandırılmış raporlara dönüştürmeyi hedefleyen yapay zekâ projesi.

An artificial intelligence project aiming to classify brain tumors from MRI images using deep learning and generate structured analysis reports to support medical professionals in reviewing model findings.

**Project Status:** In Development

**Project Areas:** Deep Learning | Medical Image Analysis | Computer Vision | Explainable AI | Clinical Decision Support

---

# Türkçe

## Proje Hakkında

Bu proje, beyin manyetik rezonans (MR) görüntülerinin derin öğrenme yöntemleriyle analiz edilmesini, olası tümör kategorilerinin sınıflandırılmasını ve elde edilen bulguların sağlık profesyonellerine anlaşılır bir rapor hâlinde sunulmasını hedeflemektedir.

Çalışma kapsamında Glioma, Meningioma, Pituitary ve No Tumor kategorilerini içeren dört sınıflı bir görüntü veri seti bulunmaktadır.

Projenin uzun vadeli amacı yalnızca bir görüntü sınıflandırma modeli geliştirmek değil, aynı zamanda sınıflandırma sonuçlarını doktorların değerlendirebileceği yapılandırılmış raporlara dönüştüren bir klinik karar destek prototipi oluşturmaktır.

Planlanan sistemde MR görüntülerinin analiz edilmesi, olası sınıf tahminlerinin oluşturulması, güven skorlarının sunulması, model kararlarının görselleştirilmesi ve analiz sonuçlarının PDF formatında raporlanması hedeflenmektedir.

Proje geliştirme aşamasındadır. Mevcut GitHub deposunda eğitim ve test veri setleri bulunmaktadır. Model eğitimi, otomatik analiz ve rapor oluşturma modülleri henüz depoda uygulanmış değildir.

## Projenin Amacı

Projenin temel amacı, beyin MR görüntülerinden elde edilen verileri yapay zekâ yöntemleriyle analiz ederek doktorların inceleyebileceği destekleyici bilgiler oluşturmaktır.

Bu doğrultuda aşağıdaki özelliklerin geliştirilmesi hedeflenmektedir:

- Beyin MR görüntülerini dört farklı kategoride sınıflandırmak.
- Görüntü ön işleme ve derin öğrenme yöntemlerini uygulamak.
- Modelin tahmin ettiği kategoriyi ve sınıf güven skorlarını sunmak.
- Açıklanabilir yapay zekâ yöntemleriyle modelin kararına katkıda bulunan görüntü bölgelerini görselleştirmek.
- Görüntü ve model sonuçlarını yapılandırılmış analiz raporlarında birleştirmek.
- Doktorların analiz sonuçlarını inceleyebileceği bir arayüz geliştirmek.
- Raporları PDF formatında oluşturmak ve saklamak.
- Doktorların kendi değerlendirmelerini ve notlarını raporlara ekleyebilmesini sağlamak.
- Geliştirilen modellerin performansını test verileri üzerinde ölçmek ve karşılaştırmak.

Sistem bağımsız tıbbi tanı koymak amacıyla değil, doktorların klinik değerlendirmesine yardımcı olabilecek araştırma amaçlı bir prototip olarak tasarlanmaktadır.

## Veri Seti

Projede beyin MR görüntülerinden oluşan dört sınıflı bir veri seti kullanılmaktadır.

### Görüntü Sınıfları

| Sınıf | Açıklama |
|---|---|
| Glioma | Glioma tümörüne ait MR görüntüleri |
| Meningioma | Meningioma tümörüne ait MR görüntüleri |
| Pituitary | Hipofiz tümörüne ait MR görüntüleri |
| No Tumor | Tümör bulunmayan MR görüntüleri |

### Veri Seti Dağılımı

| Sınıf | Eğitim | Test |
|---|---:|---:|
| Glioma | 1.321 | 300 |
| Meningioma | 1.339 | 306 |
| No Tumor | 1.595 | 405 |
| Pituitary | 1.457 | 300 |
| Toplam | 5.712 | 1.311 |

**Toplam görüntü sayısı: 7.023**

Veri seti eğitim ve test görüntüleri olarak ayrılmıştır. Her sınıfa ait görüntüler ayrı klasörlerde tutulmaktadır.

## Planlanan Sistem Mimarisi

Sistemin aşağıdaki temel bileşenlerden oluşması hedeflenmektedir.

### 1. MR Görüntüsü Yükleme

Kullanıcının analiz edilmesini istediği MR görüntüsünü sisteme yükleyebilmesi planlanmaktadır.

Yüklenen görüntü, modelin giriş gereksinimlerine uygun şekilde işlenecektir.

### 2. Görüntü Ön İşleme

Model eğitiminde ve tahmin sürecinde kullanılacak görüntüler üzerinde gerekli ön işleme işlemleri gerçekleştirilecektir.

Planlanan işlemler:

- Görüntülerin yeniden boyutlandırılması
- Piksel değerlerinin uygun biçimde normalleştirilmesi
- Eğitim verileri için veri artırma yöntemlerinin değerlendirilmesi
- Model giriş formatının hazırlanması

### 3. Derin Öğrenme ile Sınıflandırma

Beyin MR görüntülerinin dört kategoride sınıflandırılması için derin öğrenme modellerinin geliştirilmesi hedeflenmektedir.

Bu aşamada CNN mimarileri ve uygun transfer öğrenme yöntemleri değerlendirilebilecektir.

Model mimarisi ve eğitim parametreleri henüz kesinleştirilmemiştir.

### 4. Açıklanabilir Yapay Zekâ

Modelin sınıflandırma kararına katkıda bulunan görüntü bölgelerini görselleştirmek amacıyla açıklanabilir yapay zekâ yöntemlerinden yararlanılması planlanmaktadır.

Örneğin Grad-CAM benzeri teknikler, modelin hangi görüntü bölgelerine odaklandığını incelemek için değerlendirilebilir.

Bu görselleştirmeler, klinik olarak doğrulanmış tümör sınırları veya kesin lezyon konumu olarak yorumlanmayacaktır.

### 5. Klinik Analiz Raporlama Modülü

Projenin önemli hedeflerinden biri, yapay zekâ modelinin ürettiği sonuçları doktorların inceleyebileceği yapılandırılmış raporlara dönüştürmektir.

Planlanan rapor içeriği:

| Rapor Alanı | Açıklama |
|---|---|
| Rapor Kimliği | Analize ait benzersiz kayıt numarası |
| Görüntü Bilgileri | Analiz edilen MR görüntüsüne ilişkin bilgiler |
| Tahmin Edilen Sınıf | Modelin en yüksek skorla tahmin ettiği kategori |
| Sınıf Skorları | Dört kategoriye ilişkin model skorları |
| Görsel Açıklama | Uygun yöntemle oluşturulan açıklanabilirlik haritası |
| Teknik Bilgiler | Kullanılan model sürümü ve analiz bilgileri |
| Doktor Notları | Sağlık profesyonelinin değerlendirme alanı |
| Rapor Tarihi | Analizin gerçekleştirildiği tarih |

Raporların PDF formatında oluşturulması ve gerektiğinde tekrar incelenebilmesi hedeflenmektedir.

Model güven skorları doğrudan klinik tanı olasılığı veya tıbbi kesinlik anlamına gelmemektedir.

### 6. Doktor Değerlendirme Arayüzü

Planlanan arayüz üzerinden doktorların:

- Orijinal MR görüntüsünü inceleyebilmesi
- Modelin sınıflandırma sonuçlarını görebilmesi
- Açıklanabilirlik görsellerini değerlendirebilmesi
- Analiz raporunu görüntüleyebilmesi
- Kendi değerlendirme notlarını ekleyebilmesi
- PDF raporlarını dışa aktarabilmesi

hedeflenmektedir.

## Planlanan Analiz Raporu Akışı

```text
MR Görüntüsü
      |
      v
Görüntü Ön İşleme
      |
      v
Derin Öğrenme Modeli
      |
      v
Sınıflandırma Sonuçları
      |
      v
Model Skorları ve Görsel Açıklamalar
      |
      v
Yapılandırılmış Analiz Raporu
      |
      v
Doktor Değerlendirmesi
      |
      v
PDF Raporu
```

Bu akış hedeflenen sistem mimarisini göstermektedir. Mevcut depoda henüz çalışan bir uçtan uca analiz sistemi bulunmamaktadır.

## Model Performansının Değerlendirilmesi

Model uygulaması tamamlandığında performansın aşağıdaki metriklerle değerlendirilmesi planlanmaktadır:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Sınıf bazlı performansın ayrıca incelenmesi ve yanlış sınıflandırılan MR görüntülerinin analiz edilmesi hedeflenmektedir.

Özellikle tıbbi görüntü sınıflandırma problemlerinde yalnızca genel doğruluk oranı yerine sınıf bazlı hata oranlarının da değerlendirilmesi önemlidir.

Henüz doğrulanmış model performans sonuçları paylaşılmamıştır.

## Mevcut Proje Yapısı

```text
MRI-BrainTumor-Project/
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── .gitignore
├── LICENSE
└── README.md
```

## Kurulum ve Veri Setine Erişim

Projeyi bilgisayarınıza indirmek için:

```bash
git clone https://github.com/sumeyyeduz35/MRI-BrainTumor-Project.git
```

Veri seti aşağıdaki klasörlerde bulunmaktadır:

- `dataset/Training`
- `dataset/Testing`

Model eğitimi, görüntü analizi ve raporlama modüllerinin çalıştırılmasına ilişkin talimatlar, ilgili bileşenler geliştirildikçe eklenecektir.

## Geliştirme Durumu

| Bileşen | Durum |
|---|---|
| Eğitim veri seti | Mevcut |
| Test veri seti | Mevcut |
| Dört sınıflı veri organizasyonu | Mevcut |
| Görüntü ön işleme modülü | Planlandı |
| Derin öğrenme modeli | Planlandı |
| Model eğitimi ve değerlendirme | Planlandı |
| Açıklanabilir yapay zekâ | Planlandı |
| Doktor değerlendirme arayüzü | Planlandı |
| PDF analiz raporu | Planlandı |
| Rapor kayıt ve görüntüleme | Planlandı |

## Tıbbi Kullanım ve Güvenlik

Bu proje araştırma, eğitim ve yazılım geliştirme amaçlıdır.

Geliştirilecek yapay zekâ sistemi bir tıbbi tanı sistemi olarak değerlendirilmemelidir.

Model çıktıları uzman hekimlerin klinik değerlendirmelerinin yerine geçmez.

Gerçek hasta verileriyle çalışılmadan önce veri gizliliği, güvenlik, gerekli izinler, klinik doğrulama ve ilgili düzenleyici gereklilikler ayrıca ele alınmalıdır.

---

# English

## About the Project

This project aims to analyze brain magnetic resonance imaging (MRI) scans using deep learning techniques, classify potential brain tumor categories, and present the resulting findings in structured reports for medical professionals.

The project uses a four-class MRI image dataset containing Glioma, Meningioma, Pituitary, and No Tumor categories.

The long-term objective is to develop not only an image classification model but also a research-oriented clinical decision support prototype that organizes model predictions into structured reports for physician review.

The planned system will include MRI image analysis, classification predictions, confidence scores, visual explanations, and PDF report generation.

The project is currently under development. The repository contains training and testing datasets, while model training, automated analysis, and reporting components have not yet been implemented in the published code.

## Project Objectives

The primary objective is to investigate deep learning methods for analyzing brain MRI images and generating supporting information for medical professionals.

Planned objectives include:

- Classifying brain MRI images into four categories
- Implementing image preprocessing and deep learning methods
- Displaying predicted classes and model confidence scores
- Visualizing image regions contributing to model predictions
- Creating structured reports containing classification results
- Developing an interface for physician review
- Exporting analysis results as PDF documents
- Allowing physicians to add notes and evaluations
- Evaluating model performance using independent test data

The proposed system is intended as a research prototype and not as an autonomous medical diagnostic tool.

## Dataset

The dataset contains brain MRI images organized into four categories.

### Classification Categories

| Class | Description |
|---|---|
| Glioma | MRI images associated with glioma tumors |
| Meningioma | MRI images associated with meningioma tumors |
| Pituitary | MRI images associated with pituitary tumors |
| No Tumor | MRI images without a labeled tumor |

### Dataset Distribution

| Class | Training | Testing |
|---|---:|---:|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| Total | 5,712 | 1,311 |

**Total images: 7,023**

The dataset is organized into training and testing directories, with each class stored in a separate folder.

## Planned System Architecture

The proposed system consists of several components.

### 1. MRI Image Upload

Users will be able to upload an MRI image for analysis.

Uploaded images will be processed to match the input requirements of the classification model.

### 2. Image Preprocessing

Preprocessing techniques will be applied to prepare MRI images for training and inference.

Planned operations include:

- Image resizing
- Pixel normalization
- Evaluation of data augmentation techniques
- Preparation of model input data

### 3. Deep Learning Classification

Deep learning models will be developed to classify brain MRI images into four categories.

CNN architectures and transfer learning approaches may be evaluated during model development.

The final architecture and training configuration have not yet been selected.

### 4. Explainable Artificial Intelligence

Explainable AI techniques will be investigated to visualize image regions that contribute to model predictions.

Methods such as Grad-CAM may be evaluated to better understand the model's classification behavior.

These visualizations will not be interpreted as clinically validated tumor boundaries or definitive lesion locations.

### 5. Clinical Analysis Reporting Module

A major objective is to transform model predictions into structured reports that medical professionals can review.

The proposed report will include:

| Report Field | Description |
|---|---|
| Report ID | Unique analysis identifier |
| Image Information | Information related to the analyzed MRI image |
| Predicted Class | Highest-scoring classification category |
| Class Scores | Model scores for the four categories |
| Visual Explanation | Explainability visualization, when available |
| Technical Information | Model version and analysis details |
| Physician Notes | Medical professional's review and comments |
| Report Date | Date of analysis |

PDF report generation and report storage are planned features.

Model confidence scores must not be interpreted as direct clinical probabilities or diagnostic certainty.

### 6. Physician Review Interface

The planned interface will allow medical professionals to:

- Review original MRI images
- Examine classification predictions
- Inspect visual explanations
- View structured analysis reports
- Add professional evaluation notes
- Export analysis results as PDF files

## Planned Analysis Workflow

```text
MRI Image
    |
    v
Image Preprocessing
    |
    v
Deep Learning Model
    |
    v
Classification Results
    |
    v
Model Scores and Visual Explanations
    |
    v
Structured Analysis Report
    |
    v
Physician Review
    |
    v
PDF Report
```

This workflow represents the proposed architecture. An end-to-end implementation is not yet available in the repository.

## Model Evaluation

Once the classification model is implemented, performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Class-specific evaluation and misclassification analysis are planned.

For medical image classification, overall accuracy alone is insufficient to establish model reliability.

No verified model performance results have been published at this stage.

## Project Structure

```text
MRI-BrainTumor-Project/
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/sumeyyeduz35/MRI-BrainTumor-Project.git
```

The dataset is available under:

- `dataset/Training`
- `dataset/Testing`

Instructions for model training, image analysis, and report generation will be added as the corresponding components are developed.

## Development Status

| Component | Status |
|---|---|
| Training dataset | Available |
| Testing dataset | Available |
| Four-class data organization | Available |
| Image preprocessing | Planned |
| Deep learning model | Planned |
| Model training and evaluation | Planned |
| Explainable AI | Planned |
| Physician review interface | Planned |
| PDF report generation | Planned |
| Report storage and retrieval | Planned |

## Medical Disclaimer

This project is intended for research, educational, and software development purposes.

The proposed AI system is not a clinically validated medical diagnostic tool.

Model predictions must not replace professional medical assessment or clinical judgment.

Before using real patient information, data privacy, security, appropriate permissions, clinical validation, and applicable regulatory requirements must be addressed.

## Project Status

**In Development**

The current repository contains the MRI dataset structure. Deep learning classification, explainable AI, and clinical reporting capabilities are planned for future development.
