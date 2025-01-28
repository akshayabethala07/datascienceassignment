from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

# Feature engineering: Total transaction value and frequency
customer_summary = customer_transactions.groupby('CustomerID').agg(
    total_spend=('TotalValue', 'sum'),
    transaction_count=('TransactionID', 'count')
).reset_index()

# Apply KMeans clustering (we will try with 4 clusters here)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_summary['Cluster'] = kmeans.fit_predict(customer_summary[['total_spend', 'transaction_count']])

# Calculate DB Index
db_index = davies_bouldin_score(customer_summary[['total_spend', 'transaction_count']], customer_summary['Cluster'])
print(f"DB Index: {db_index}")

# Visualize the clusters using PCA (reduce to 2D)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(customer_summary[['total_spend', 'transaction_count']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=customer_summary['Cluster'], palette='Set2')
plt.title('Customer Segmentation (K-Means Clustering)')
plt.show()

# Save clustering results
customer_summary[['CustomerID', 'Cluster']].to_csv('Clusters.csv', index=False)
