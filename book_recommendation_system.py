import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template

app = Flask(__name__)

# Function to fetch book data from Google Books API
def fetch_books(query='programming'):
    API_URL = 'https://www.googleapis.com/books/v1/volumes'
    params = {
        'q': query,
        'maxResults': 40,
        'key': 'YOUR_GOOGLE_BOOKS_API_KEY'  # Replace with your API key
    }
    response = requests.get(API_URL, params=params)
    books = response.json()['items']
    book_list = []

    for book in books:
        book_info = book['volumeInfo']
        title = book_info.get('title', 'No title')
        authors = ", ".join(book_info.get('authors', ['Unknown']))
        description = book_info.get('description', '')
        book_list.append({
            'title': title,
            'authors': authors,
            'description': description
        })

    return pd.DataFrame(book_list)

# Function for Content-Based Filtering
def content_based_filtering(books_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

# Function to get recommendations based on book description (content-based)
def get_content_based_recommendations(book_title, books_df, cosine_sim):
    idx = books_df.index[books_df['title'] == book_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations

    recommended_books = []
    for i in sim_scores:
        recommended_books.append(books_df['title'].iloc[i[0]])

    return recommended_books

# Function for Collaborative Filtering (using K-Nearest Neighbors)
def collaborative_filtering(books_df):
    # We will simulate user ratings for simplicity
    ratings = pd.DataFrame({
        'user1': [5, 4, 3, 2, 1],
        'user2': [3, 4, 4, 5, 2],
        'user3': [1, 5, 5, 3, 4],
        'user4': [2, 3, 4, 4, 5],
        'user5': [4, 3, 2, 5, 4]
    }, index=books_df['title'])

    model_knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    model_knn.fit(ratings.T)  # We transpose to make books the columns
    return model_knn

# Function to get recommendations using Collaborative Filtering
def get_collaborative_recommendations(book_title, books_df, model_knn, ratings):
    book_idx = books_df.index[books_df['title'] == book_title].tolist()[0]
    book_ratings = ratings.iloc[book_idx].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(book_ratings)

    recommended_books = []
    for idx in indices[0]:
        recommended_books.append(books_df['title'].iloc[idx])

    return recommended_books

# Web interface to interact with the recommendation system
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    books_df = fetch_books(query=query)
    
    cosine_sim = content_based_filtering(books_df)
    content_recommendations = get_content_based_recommendations('Dune', books_df, cosine_sim)
    
    ratings = pd.DataFrame({
        'user1': [5, 4, 3, 2, 1],
        'user2': [3, 4, 4, 5, 2],
        'user3': [1, 5, 5, 3, 4],
        'user4': [2, 3, 4, 4, 5],
        'user5': [4, 3, 2, 5, 4]
    }, index=books_df['title'])

    model_knn = collaborative_filtering(books_df)
    collaborative_recommendations = get_collaborative_recommendations('Dune', books_df, model_knn, ratings)
    
    return render_template('recommendations.html', content_recommendations=content_recommendations, collaborative_recommendations=collaborative_recommendations)

if __name__ == "__main__":
    app.run(debug=True)
