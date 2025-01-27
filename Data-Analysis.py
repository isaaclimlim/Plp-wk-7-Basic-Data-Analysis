import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    # Load the dataset
    df = pd.read_csv('iris.csv', header=None, names=[
        'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'
    ])
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(df.head())
    
    # Explore the structure of the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Clean the dataset (fill missing values with mean, though the Iris dataset has none)
    df = df.fillna(df.mean())
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
except FileNotFoundError:
    print("Error: The dataset file was not found. Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by Species and compute the mean of numerical columns
grouped = df.groupby('Species').mean()
print("\nMean of numerical columns grouped by Species:")
print(grouped)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart showing trends over time (simulated with PetalLength as an example)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df.reset_index(), x='index', y='PetalLength', hue='Species', markers=True)
plt.title('Petal Length Trends')
plt.xlabel('Index (Simulating Time)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Bar chart showing average Sepal Width per Species
plt.figure(figsize=(10, 6))
sns.barplot(x='Species', y='SepalWidth', data=df, estimator='mean')
plt.title('Average Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Histogram of Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(df['SepalLength'], kde=True, bins=20)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Sepal Length vs. Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SepalLength', y='PetalLength', hue='Species', style='Species', data=df)
plt.title('Relationship Between Sepal Length and Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

print("\nScript execution complete. All tasks have been performed.")
