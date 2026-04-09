# K-Means From Scratch

This repository contains my implementation of the **K-Means clustering algorithm from scratch using NumPy**.

The goal of this project was not only to obtain working clusters, but to **understand how the algorithm works internally**, implementing each step manually and experimenting with improvements along the way.

---

# Project Overview

K-Means is an **unsupervised machine learning algorithm** that groups data into **K clusters** based on similarity.

The algorithm iteratively performs three main steps:

1. Initialize cluster centroids  
2. Assign each data point to the nearest centroid  
3. Update the centroid positions using the mean of the assigned points  

This process repeats until the centroids stop changing.

The algorithm minimizes the following objective function:

$$
J = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2
$$

Where:

- $x_i$ is a data point
- $\mu_{c_i}$ is the centroid of the cluster assigned to that point
- $n$ is the number of samples

This value is known as **inertia**, which represents how compact the clusters are.

---

## Dataset

The dataset used in this project is the **Mall Customers Dataset**, a small dataset commonly used for learning and demonstrating clustering algorithms such as K-Means.

It contains **200 samples and 5 features** describing mall customers:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

The **Spending Score** represents a value assigned by the mall based on customer behavior and purchasing patterns.

This dataset is widely used for **customer segmentation tasks**, where clustering algorithms help identify groups of customers with similar spending habits or income profiles.

### Source

The dataset used in this repository was loaded from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python), originally known as the **Mall Customers Dataset**.

### License

The dataset is generally distributed under **public or permissive licenses** and is widely used for educational and demonstration purposes in machine learning projects.

---

# Development Process

This project was built iteratively. Instead of implementing everything at once, I improved the algorithm step by step and analyzed the results through visualizations.

---

# First Attempt

The first version of the algorithm implemented the basic K-Means procedure:

- Random centroid initialization
- Distance calculation using Euclidean distance
- Cluster assignment
- Centroid update

However, the clustering results were **incorrect**, producing unstable cluster shapes.

### Result

![First Attempt](static/K-means%20Clusters(error).png)

---

# Data Normalization

Since K-Means relies on **distance calculations**, feature scale can strongly influence the results.

To address this, I applied **Min-Max normalization** to scale the features between 0 and 1:

$$
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Even after normalizing the data, the clustering results were still not correct.

*(Unfortunately I lost the visualization produced at this stage.)*

---

# Adding Inertia Calculation

At this point, I implemented the **inertia metric**, which calculates the total squared distance between each point and its assigned centroid.

This allowed the algorithm to:

- Evaluate clustering quality
- Compare different model initializations (`n_init`)

After adding inertia, the algorithm began producing **correct clusters**.

---

# Results (Normalized Data)

With normalized data, the clustering produced well-separated groups.

![Normalized Clusters](static/K-means%20Clusters%20Normalized.png)

---

# Results (Original Data)

The algorithm also worked correctly with the **original dataset**.

![Original Data Clusters](static/K-means%20Clusters.png)

---

# Attempting the Elbow Method

Next, I tried implementing the **Elbow Method** to automatically determine the optimal number of clusters.

The idea is to compute inertia for multiple values of K and look for the point where the improvement starts diminishing.

However, my automatic elbow detection approach **did not work reliably**.

### Elbow Attempt

![Elbow Graph](static/K-means%20Clusters%20Normalized%20Elbow.png)

---

# Next Steps

Future improvements planned for this project include:

- Implementing a more reliable method for selecting **K**
- Trying alternative cluster validation metrics such as:
  - Silhouette Score
  - Gap Statistic
- Improving performance through **NumPy vectorization**

---

## Project Structure
```
kmeans-from-scratch/
├── src/
│   └── kmeans.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
└── README.md
```

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Motivation

This project was created as a way to deepen my understanding of **unsupervised learning algorithms** by implementing them from scratch rather than relying on existing libraries.

Building algorithms manually helps develop intuition about:

- optimization
- clustering behavior
- numerical stability
- model evaluation

---

# Future Work

Possible extensions of this project include:

- implementing **K-Means with full NumPy vectorization**
- comparing results with the implementation from `scikit-learn`
- implementing other clustering algorithms such as:
  - DBSCAN
  - Hierarchical Clustering
  - Gaussian Mixture Models

---

# Author

Artur Catão