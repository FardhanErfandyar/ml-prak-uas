import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import data


df = pd.read_csv("Airline_customer_satisfaction.csv")
df.head()

# amati bentuk data
df.shape
print(df.shape)

# Melihat ringkasan statistik deskriptif dari DataFrame
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

# Assuming 'df' is your DataFrame
plt.figure(figsize=(14, 8))  # Adjusting the figure size for better visibility
columns = [
    "Age",
    "Flight Distance",
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]
sns.boxplot(data=df[columns])
plt.title("Boxplot of Selected Features")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
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
# Assuming 'df' is your DataFrame
plt.figure(figsize=(14, 8))  # Adjusting the figure size for better visibility
columns = [
    "Age",
    "Flight Distance",
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]
sns.boxplot(data=df[columns])
plt.title("Boxplot of Selected Features")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# Menampilkan data setelah menangani outlier
print("Data setelah menangani outlier".center(75, "="))
print(df.head())
print("=" * 75)

# amati bentuk visual masing-masing fitur
plt.style.use("fivethirtyeight")
plt.figure(1, figsize=(15, 6))
n = 0
for x in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)
    plt.title("Distplot of {}".format(x))

plt.show()

# Ploting untuk mencari relasi antara Age , Annual Income and Spending Score
plt.figure(1, figsize=(15, 20))
n = 0
for x in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    for y in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(x=x, y=y, data=df)
        plt.ylabel(y.split()[0] + " " + y.split()[1] if len(y.split()) > 1 else y)

plt.show()

# Melihat sebaran Food and drink dan Arrival Delay in Minutes terhadap Customer type
plt.figure(1, figsize=(15, 8))
for CustomerType in ["Loyal Customer", "disloyal Customer"]:
    plt.scatter(
        x="Departure Delay in Minutes",
        y="Arrival Delay in Minutes",
        data=df[df["Customer Type"] == CustomerType],
        s=200,
        alpha=0.5,
        label=CustomerType,
    )
    plt.xlabel("Departure Delay in Minutes"), plt.ylabel("Arrival Delay in Minutes")
    plt.title("Departure Delay in Minutes vs Arrival Delay in Minutes")
    plt.legend()
plt.show()


# Merancang K-Means untuk Food and drink dan Food and drink
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[["Departure Delay in Minutes", "Arrival Delay in Minutes"]].iloc[:, :].values
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
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

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
plt.scatter(
    x="Departure Delay in Minutes",
    y="Arrival Delay in Minutes",
    data=df,
    c=labels2,
    s=200,
)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c="red", alpha=0.5)
plt.ylabel("Departure Delay in Minutes"), plt.xlabel("Arrival Delay in Minutes)")
plt.show()


# Melihat nilai Silhouette Score
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)
