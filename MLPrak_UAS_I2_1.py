# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import data
print("Data apple quality".center(75, "="))
df = pd.read_csv("Airline_customer_satisfaction.csv")
df.head()
print(df.head())

# amati bentuk data
print("Bentuk Data".center(75, "="))
df.shape
print(df.shape)

# Melihat ringkasan statistik deskriptif dari DataFrame
print("Statistika deskriptif dari dataframe".center(75, "="))
df.describe()
print(df.describe())

# cek null data
# Pengecekan missing value
print("Pengecekan missing value".center(75, "="))
print(df.isnull().sum())
print("=" * 75)

# Penanganan Missing value (Menghapus baris yang mengandung nilai null)
df.dropna(inplace=True)


# Menampilkan Data setelah menghapus missing value
print("Data setelah menghapus missing value".center(75, "="))
print(df.isnull().sum())
print("=" * 75)

# cek outlier
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[
        [
            "epalLengthCm",
            "Weight",
            "Sweetness",
            "Crunchiness",
            "Juiciness",
            "Ripeness",
            "Acidity",
        ]
    ]
)
plt.title("Boxplot")
plt.show()


# Penanganan outliers dengan IQR (Interquartile Range)
for column in df._get_numeric_data().columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where(
        df[column] < lower_bound,
        lower_bound,
        np.where(df[column] > upper_bound, upper_bound, df[column]),
    )
# cek outlier
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[
        [
            "Size",
            "Weight",
            "Sweetness",
            "Crunchiness",
            "Juiciness",
            "Ripeness",
            "Acidity",
        ]
    ]
)
plt.title("Boxplot")
plt.show()


# Menampilkan data setelah menangani outlier
print("Data setelah menangani outlier".center(75, "="))
print(df.head())
print("=" * 75)


# amati bentuk visual masing-masing fitur
plt.style.use("fivethirtyeight")
plt.figure(figsize=(10, 6))
n = 0
for x in [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
]:
    n += 1
    plt.subplot(1, 7, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)

plt.show()

# Ploting untuk mencari relasi antara Size dan Sweetness terhadap Quality
plt.figure(1, figsize=(15, 20))
numeric_cols = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
]
n = 0
for x in numeric_cols:
    for y in numeric_cols:
        n += 1
        plt.subplot(7, 7, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        # Convert x and y to numeric if they are not already
        x_data = pd.to_numeric(df[x], errors="coerce")
        y_data = pd.to_numeric(df[y], errors="coerce")

        sns.regplot(x=x_data, y=y_data, data=df)
        plt.ylabel(y)

plt.show()


# Melihat sebaran Size dan Sweetness terhadap Quality
plt.figure(1, figsize=(15, 8))
for Quality in ["good", "bad"]:
    plt.scatter(
        x="Size",
        y="Sweetness",
        data=df[df["Quality"] == Quality],
        s=200,
        alpha=0.5,
        label=Quality,
    )
    plt.xlabel("Size"), plt.ylabel("Sweetness")
    plt.title("Size vs Sweetness")
    plt.legend()
plt.show()


# Merancang K-Means untuk Size dan Sweetness
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[["Size", "Sweetness"]].iloc[:, :].values
inertia = []
for n in range(1, 9):
    algorithm = KMeans(
        n_clusters=n, init="k-means++", n_init=10, max_iter=300, random_state=111
    )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

# Plot bentuk visual elbow
plt.figure(1, figsize=(15, 6))
plt.plot(range(1, 9), inertia, "o")
plt.plot(range(1, 9), inertia, "-", alpha=0.5)
plt.xlabel("Number of Clusters"), plt.ylabel("Inertia")
plt.show()

# Membangun K-Means
algorithm = KMeans(
    n_clusters=2,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm="elkan",
)
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

# Menyiapkan data untuk bentuk visual cluster
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
step = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])  # array diratakan 1D


# Melihat bentuk visual cluster

plt.figure(1, figsize=(15, 7))
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(
    Z1,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Pastel2,
    aspect="auto",
    origin="lower",
)
plt.scatter(x="Size", y="Sweetness", data=df, c=labels2, s=200)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c="red", alpha=0.5)
plt.ylabel("Size"), plt.xlabel("Sweetness")
plt.show()


# Melihat nilai Silhouette Score
print("Silhouetter Score".center(75, "="))
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)
