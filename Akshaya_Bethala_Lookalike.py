from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Merge customer data with transactions
customer_transactions = transactions.merge(customers, on="CustomerID")

# Calculate total spend and frequency per customer
customer_summary = customer_transactions.groupby('CustomerID').agg(
    total_spend=('TotalValue', 'sum'),
    transaction_count=('TransactionID', 'count')
).reset_index()

# Create a customer profile matrix with total spend and transaction count
profile_matrix = customer_summary[['CustomerID', 'total_spend', 'transaction_count']]

# Compute cosine similarity between customers
similarity_matrix = cosine_similarity(profile_matrix[['total_spend', 'transaction_count']])

# Get top 3 most similar customers for each customer
top_lookalikes = {}
for idx, customer_id in enumerate(profile_matrix['CustomerID']):
    similar_indices = np.argsort(similarity_matrix[idx])[::-1][1:4]  # Top 3 excluding the customer itself
    top_lookalikes[customer_id] = [(profile_matrix['CustomerID'][i], similarity_matrix[idx][i]) for i in similar_indices]

# Display the results in the expected format
lookalike_results = []
for cust_id, lookalikes in top_lookalikes.items():
    for lookalike, score in lookalikes:
        lookalike_results.append([cust_id, lookalike, score])

lookalike_df = pd.DataFrame(lookalike_results, columns=['CustomerID', 'LookalikeID', 'SimilarityScore'])
lookalike_df.to_csv('Lookalike.csv', index=False)
