# Customer Segmentation Using Clustering

## Introduction
Customer segmentation is essential for understanding customer behavior and implementing effective marketing strategies. This project uses RFM (Recency, Frequency, Monetary) analysis and KMeans clustering to segment customers based on their purchasing behavior. The goal is to identify distinct customer groups to tailor marketing efforts and improve business outcomes. The dataset used contains various transaction details to analyze customer behavior.

## Data Preparation
The dataset contains columns such as `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`. We perform the following steps to prepare the data:

1. **Load the Dataset:** Read the data from a CSV file.
2. **Data Cleaning:** Convert `InvoiceDate` to datetime format and calculate `TotalPrice` for each transaction.
3. **RFM Calculation:** Compute Recency, Frequency, and Monetary metrics for each customer.

```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv', encoding='ISO-8859-1')

# Data cleaning and preprocessing
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Calculate RFM metrics
snapshot_date = data['InvoiceDate'].max() + pd.DateOffset(days=1)
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

Methodology
Scaling the Data

We scale the RFM metrics to ensure that each metric contributes equally to the clustering process.

python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

KMeans Clustering

We use KMeans clustering to segment customers into distinct groups.

python

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_

Visualizing the Clusters

We visualize the clusters using a 3D scatter plot to better understand the distribution of customers.

python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm_scaled[:, 0], rfm_scaled[:, 1], rfm_scaled[:, 2], c=rfm['Cluster'], cmap='viridis')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('3D Scatter Plot of Clusters')
fig.colorbar(scatter, ax=ax, label='Cluster')
plt.show()

Results
Cluster Profiles and Business Insights

python

print("\n### Cluster Profiles and Business Insights ###\n")

# Cluster 0
print("Cluster 0:")
print(" - Average Recency: 2.33 (higher than average)")
print(" - Average Frequency: -0.43 (lower than average)")
print(" - Average Monetary: 8.36 (significantly higher than average)")
print(" - Insights: These customers are recent and very high spenders, despite a lower purchase frequency. Focus on retaining them with exclusive offers.\n")

# Cluster 1
print("Cluster 1:")
print(" - Average Recency: -0.91 (lower than average)")
print(" - Average Frequency: 0.35 (slightly higher than average)")
print(" - Average Monetary: 0.25 (slightly higher than average)")
print(" - Insights: These are moderately engaged customers with a reasonable frequency and monetary value. Consider loyalty programs to boost their spending.\n")

# Cluster 2
print("Cluster 2:")
print(" - Average Recency: -0.18 (close to average)")
print(" - Average Frequency: -0.04 (close to average)")
print(" - Average Monetary: -0.03 (close to average)")
print(" - Insights: These customers are average in all aspects. General marketing strategies should work well.\n")

# Cluster 3
print("Cluster 3:")
print(" - Average Recency: -0.74 (lower than average)")
print(" - Average Frequency: -0.43 (lower than average)")
print(" - Average Monetary: -0.03 (close to average)")
print(" - Insights: These are at-risk customers who may need re-engagement strategies such as special offers or reminders to encourage purchases.\n")

# Cluster 4
print("Cluster 4:")
print(" - Average Recency: 2.17 (higher than average)")
print(" - Average Frequency: -0.43 (lower than average)")
print(" - Average Monetary: -0.19 (lower than average)")
print(" - Insights: These customers are recent but spend less frequently and have lower monetary value. They might respond well to targeted promotions to increase spending.\n")

Recommendations

Based on the cluster profiles, the following actions are recommended:

    Cluster 0: Focus on exclusive offers to retain these high-value customers.
    Cluster 1: Implement loyalty programs to increase spending among these moderately engaged customers.
    Cluster 2: General marketing strategies should suffice as these customers are average in all aspects.
    Cluster 3: Use re-engagement strategies such as special offers or reminders to bring back these at-risk customers.
    Cluster 4: Targeted promotions might help increase the frequency and monetary value of these recent but less engaged customers.

Conclusion

This project demonstrates the application of RFM analysis and KMeans clustering for customer segmentation. By identifying and understanding different customer segments, businesses can tailor their marketing strategies to improve customer engagement and drive sales.
Requirements

    Python 3.x
    pandas
    numpy
    scikit-learn
    matplotlib

How to Run

    Ensure you have all the required libraries installed.
    Download the dataset and place it in the same directory as the script.
    Run the script to see the customer segmentation and business insights.

Author

Krishna Acharya


