import pandas as pd

data = pd.read_csv("netflix_titles.csv")

print("Dataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nMovies vs TV Shows:")
print(data['type'].value_counts())

print("\nTop Countries:")
print(data['country'].value_counts().head(10))