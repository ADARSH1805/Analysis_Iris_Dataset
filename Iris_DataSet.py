import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# This script loads the Iris dataset and performs basic data exploration.
# Load the Iris dataset
data = pd.read_csv("D:/45 Days Data Science/Datasets/Iris_dataset.csv")
print("Data loaded successfully.")
#glimpse of whole data
print(data)

#Extract information about the dataset
print("\nDataset Information:")
print(data.info())

#extract statistical information - numerical columns
print("\nStatistical Summary:")
print(data.describe())

# Extract the first 5 rows of the dataset
print("\nFirst 5 Rows:")
print(data.head())
# Extract the last 5 rows of the dataset
print("\nLast 5 Rows:")
print(data.tail())

# Column names
print("\nColumn Names:")
print(data.columns)
# Column name in Vertical Format
print("\nColumn Names in Vertical Format:")
for column in data.columns:
    print(column)
    
# Shape of the dataset
print("\nShape of the Dataset:")
print(data.shape)

# Statistical Functions
print("\nStatistical Functions:")
print("Mean: ", data.mean(numeric_only=True))
print("Median: ", data.median(numeric_only=True))
print("Mode: ", data.mode(numeric_only=True).iloc[0])  # Mode returns a DataFrame, take the first row
print("Standard Deviation: ", data.std(numeric_only=True))
print("Variance: ", data.var(numeric_only=True))
print("Minimum: ", data.min(numeric_only=True))
print("Maximum: ", data.max(numeric_only=True))

# Unique values in each column
print("\nUnique Values in Each Column:")
print(data.nunique())
#list the unique value present
print("\nUnique Species:")
print(data['Species'].unique())

# Rename Columns Name
data.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width'
}, inplace=True)
print("\nRenamed Columns:")
print(data.columns)

#check for missing values
print("\nMissing Values in the Dataset:")
print(data.isnull())
# Count missing values in each column
print("\nCount of Missing Values in Each Column:")
print(data.isnull().sum())

# check for duplicate values
print("\nDuplicate Rows in the Dataset:")
print(data.duplicated())
# Count of duplicate rows
print("\nCount of Duplicate Rows:")
print(data.duplicated().sum())


# Visualize the distribution of each feature

#Univariate Visualization
# countplot for species
# Visualizing the count of each species in the dataset
print("\nCount of each Species:")
sns.countplot(x='Species', data=data, palette='viridis', hue='Species')
plt.title("Species distribution:")
plt.show()

# Box-Plot
# Visualizing how species are distributed across sepal length
print("\nBox Plot for Sepal Length by Species:")
sns.boxplot(x='Species', y='sepal_length', data=data, palette='viridis', hue='Species')
plt.title("Box Plot of Sepal Length by Species")
plt.show()

# Voiolin Plot
# Visualizing how species are distributed across petal length
print("\nViolin Plot for Petal Length by Species:")
sns.violinplot(x='Species', y='petal_length', data=data, palette='viridis', hue='Species')
plt.title("Violin Plot of Petal Length by Species")
plt.show()

# Multivariate Visualization
# Scatter Plot
# Visualizing the relationship between sepal length vs petal length
print("\nScatter Plot of Sepal Length vs Petal Length:")
sns.scatterplot(x='sepal_length', y='petal_length', data=data, hue='Species')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.show()

# Visualizing the relationship between sepal width vs petal width
print("\nScatter Plot of Sepal Width vs Petal Width:")
sns.scatterplot(x='sepal_width', y='petal_width', data=data, hue='Species')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Width vs Petal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()

# Pair Plot
# Visualizing the pairwise relationships in the dataset
print("\nPair Plot of the Dataset:")
sns.set(style="whitegrid")
sns.pairplot(data, hue='Species', height=2)
plt.title("Pair Plot of Iris Dataset")
plt.show()

# Correlation Matrix
# Visualizing the correlation between features
print("\nCorrelation Matrix:")
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Visualizing the correlation matrix using a heatmap
print("\nHeatmap of Correlation Matrix:")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features in Iris Dataset")
plt.show()

# Outliers
print("\nFinding Outliers in sepal width using IQR method:")
Q1 = data['sepal_width'].quantile(0.25)
Q3 = data['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['sepal_width'] < lower_bound) | (data['sepal_width'] > upper_bound)]
print("Outliers in sepal width:")
print(outliers)

# Visualizing the outliers in sepal width
print("\nBox Plot of Sepal Width with Outliers:")
sns.boxplot(x=data['sepal_width'])
plt.title("Box Plot of Sepal Width with Outliers")
plt.show()

# Removing Outliers
print("\nRemoving Outliers from the Dataset:")
upper = np.where(data['sepal_width'] >= (Q3 + 1.5 * IQR))
lower = np.where(data['sepal_width'] <= (Q1 - 1.5 * IQR))
print("Shape of the dataset before removing outliers:", data.shape)
data.drop(upper[0], inplace=True)
data.drop(lower[0], inplace=True)
print("Shape of the dataset after removing outliers:", data.shape)

