<div align="center">

# Supervised Learning

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.17-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14-00897B?logo=python&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-1.5-4B8BBE?logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=matplotlib&logoColor=white)
![cloudpickle](https://img.shields.io/badge/cloudpickle-3.1-6E4C8E?logo=python&logoColor=white)
![ipykernel](https://img.shields.io/badge/ipykernel-6.29-F37626?logo=jupyter&logoColor=white)

</div>

## 📘 Deskripsi

Di sini, kami membuat package yang memuat beberapa algoritma _supervised learning_ klasik untuk masalah klasifikasi. Package ini meliputi meliputi algoritma $k$-_nearest neighbor_, _decision tree_, _logistic regression_, _support vector machine_, dan _naive bayes_. Implementasi modular berbasis kelas, menyerupai `scikit-learn`, sehingga penggunaannya mudah tetapi sangat _customizable_.

Selanjutnya, kami juga menerapkan package ini dalam menyelesaikan masalah klasifikasi biner pembatalan pemesanan hotel. Dataset diambil dari kaggle [Tugas Besar 2 IF3170](https://www.kaggle.com/competitions/T2-IF3170-2026-2027/overview). Dilakukan _exploratory data analysis_, pengolahan data (berupa _cleaning_, _transformation_, _selection_, dan _dimensionality reduction_), pemodelan (dengan setiap algoritma klasik yang telah dibuat), dan validasi.

Untuk melihat analisis lanjut mengenai implementasi package from scratch dan hasil penerapannya pada masalah klasifikasi aktual, dapat dicek laporan di [IF3170_G32_PasukanPramaPutra.pdf](./docs/IF3170_G32_PasukanPramaPutra.pdf).

---

## ✨ Fitur Utama

- 🥇 **Algoritma _Supervised Learning_**
   - $k$-_Nearest Neighbor_: berbagai pilihan metrik dan metode _voting_
   - _Decision Tree_: standar C4.5 dengan banyak opsi _loss_, metode _pruning_, dan visualisasi pohon
   - _Logistic Regression_: fitur _gradient ascent_, regularisasi _elastic net_, dan visualisasi pelatihan
   - _Support Vector Machine_: multi-kelas (OvO dan OvR), algoritma SMO, dan visualisasi garis pembagi
   - _Naive Bayes_: aturan Bayes, Laplacian smoothing, dan visualisasi Bayesian Network
- 🥈 **Klasifikasi Pembatalan Pemesanan Hotel**
   - Pengolahan Data: _data cleaning_, _data transformation_, _data selection_, dan _dimensionality reduction_
   - Pemodelan dan Validasi: menerapkan algoritma _supervised learning_ dan memvalidasi error
   - Perbandingan: membandingkan dengan `scikit-learn`

---

## 📁 Struktur Folder

```
IF3170_G32_PasukanPramaPutra/
├── doc/
│   └── IF3170_G32_PasukanPramaPutra.pdf
├── output/
│   └── submission.csv
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├─ models/
│   │  ├── base.py
│   │  ├── knn.py
│   │  ├── decision-tree.py
│   │  ├── logistic-regression.py
│   │  ├── svm.py
│   │  └── naive-bayes.py
│   ├─ eda.ipynb
│   └─ main.ipynb
└── README.md
```

---

## ⚙️ Requirement & Instalasi

### Prasyarat Package

- numpy >= 2.2.6
- pandas >= 2.3.0
- scipy >= 1.17.1
- scikit-learn >= 1.8.0
- imbalanced-learn >= .14.2
- joblib >= 1.5.1
- matplotlib >= 3.10.3
- cloudpickle >= 3.1.3
- ipykernel >= 6.29.5

### Instalasi Package

1. Clone repository.

   ```bash
   git clone https://github.com/adielrum/IF3170_G32_PasukanPramaPutra.git
   cd IF3170_G32_PasukanPramaPutra
   ```

2. Pasang dependensi.

   ```bash
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` belum ada, bisa install manual.

   ```bash
   pip install numpy pandas scipy scikit-learn imbalanced-learn joblib matplotlib cloudpickle ipykernel
   ```

---

## 🚀 Menggunakan Package

Setiap algoritma _supervised learning_ yang diimplementasikan dapat dipakai dengan mudah sebagai berikut. Metode `fit()` dan `predict()` digunakan untuk melakukan pelatihan dan klasifikasi.

### kNN

Algoritma $k$-_nearest neighbors_ memiliki hyperparameter utama $k$, metrik jarak, dan metode _voting_.

```python
from models.knn import KNN

knn = KNN(n_neighbors=5, metric="euclidean", voting_strategy="majority")

knn.fit(X_train, y_train)
y_test = knn.predict(X_test)
```

### Decision Tree

Algoritma _decision tree_ memiliki hyperparameter utama _criterion_ (fungsi _loss_) dan penggunaan _gain ratio_. Hyperparameter lainnya berkaitan dengan _pre-pruning_ (seperti kedalaman maksimum atau jumlah simpul minimum untuk percabangan) atau _post-pruning_ (seperti $\alpha$ untuk _minimal cost-complexity pruning_).

```python
from models.decision_tree import DecisionTree

dt = DecisionTree(criterion="entropy", use_gain_ratio=True)

dt.fit(X_train, y_train)
y_test = dt.predict(X_test)
```

## Logistic Regression

Algoritma _logistic regression_ memiliki hyperparameter utama _learning rate_, jumlah iterasi maksimal, dan ukuran _batch_. Hyperparameter lainnya berkaitan dengan regularisasi (penalti L1 dan L2) atau algoritma optimisasi (jenis _optimizer_ dan parameter mereka).

```python
from models.logistic_regression import LogisticRegression

lr = LogisticRegression(learning_rate=0.01, n_iter=1000, batch_size=None)

lr.fit(X_train, y_train)
y_test = lr.predict(X_test)
```

## Support Vector Machines

Algoritma _support vector machine_ memiliki hyperparameter utama regularisasi $C$ dan jenis kernel. Hyperparameter lainnya adalah parameter untuk kernel (seperti koefisien dan pangkat) atau algoritma optimisasi (toleransi minimum dan iterasi maksimum).

```python
from models.svm import SVC

svc = SVC(C=1.0, kernel="linear")

svc.fit(X_train, y_train)
y_test = svc.predict(X_test)
```

## Naive Bayes

Algoritma _naive bayes_ memiliki hyperparameter utama _Laplace smoothing_ $\alpha$ dan jumlah _binning_ untuk diskritisasi fitur. Hyperparameter lainnya hanya metode pembobotan.

```python
from models.naive_bayes import NaiveBayes

nb = NaiveBayes(alpha=1.0, n_bins=3)

nb.fit(X_train, y_train)
y_test = nb.predict(X_test)
```

---

## 👨‍💻 Authors

| Nama                   | NIM      | Pembagian Tugas                                                            |
| ---------------------- | -------- | -------------------------------------------------------------------------- |
| Adiel Rum              | 10123004 | kNN, optimisasi, bonus, laporan                                            |
| Timothy Niels Ruslim   | 10123053 | decision tree, logistic regression, SVM, laporan                           |
| Qodri Azkarayan        | 10823501 | naive bayes, laporan                                                       |
| Mohammad Najmutstsaqib | 13223040 | EDA, data preprocessing, pengujian model, improvement, submission, laporan |
| Hafizh Hananta Akbari  | 13522132 | –                                                                          |

Bukti dapat dilihat dari sejarah _commits_.

---

## 🔗 Tautan

- 📂 [Repository GitHub](https://github.com/adielrum/IF3170_G32_PasukanPramaPutra)

---

> Dibuat sebagai bagian dari Tugas Besar 2 IF3170 Inteligensi Artifisial 2026 – ITB
