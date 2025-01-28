import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Convert 'SignupDate' and 'TransactionDate' to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'], format='%d-%m-%Y')
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], format='%d-%m-%Y %H:%M')

# EDA: 1. Customers by Region
plt.figure(figsize=(8, 6))
sns.countplot(data=customers, x='Region', palette='Set2')
plt.title('Customer Distribution by Region')
plt.show()

# EDA: 2. Product Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(products['Price'], bins=30, kde=True)
plt.title('Product Price Distribution')
plt.show()

# EDA: 3. Transaction Frequency Over Time
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
transaction_counts = transactions.groupby('Month').size()

plt.figure(figsize=(10, 6))
transaction_counts.plot(kind='line')
plt.title('Transaction Frequency Over Time')
plt.ylabel('Number of Transactions')
plt.xlabel('Month')
plt.show()
