import pandas as pd

data = pd.read_csv("netflix_titles.csv")

def recommend(title):

    title = title.lower()

    matches = data[data['title'].str.lower().str.contains(title, na=False)]

    if len(matches) == 0:
        print("No similar titles found")
    else:
        print(matches[['title','type','country','release_year']].head(5))


user_input = input("Enter movie/show name: ")

recommend(user_input)