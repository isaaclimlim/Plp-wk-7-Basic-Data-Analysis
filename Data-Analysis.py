import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
try:
    # Load the dataset
    df = pd.read_csv('iris.csv', header=None, names=[
        'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class'
    ])
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(df.head())
    
    # Check the structure of the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Clean the dataset (if there were missing values)
    df = df.fillna(df.mean())
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

except FileNotFoundError:
    print("Error: The dataset file was not found. Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Data Visualization
sns.set(style="whitegrid")

# Histogram of Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(df['SepalLength'], kde=True, bins=30)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Bar chart of mean Petal Length for each class
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='PetalLength', data=df, estimator='mean')
plt.title('Mean Petal Length for Each Class')
plt.xlabel('Iris Class')
plt.ylabel('Mean Petal Length (cm)')
plt.show()

# Scatter plot of Sepal Length vs Sepal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SepalLength', y='SepalWidth', hue='Class', data=df)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

print("\nScript execution complete. All tasks have been performed.")

