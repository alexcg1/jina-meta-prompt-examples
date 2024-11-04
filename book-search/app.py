import json
import os
import requests
import numpy as np
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
from rich.progress import track
from rich.console import Console

# Ensure the Jina AI API key is set as an environment variable
JINA_API_KEY = os.getenv("JINA_API_KEY")
console = Console()

# Get your Jina AI API key for free: https://jina.ai/?sui=apikey

def load_authors(filename="authors.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        authors = [line.strip() for line in file if line.strip()]
    return authors

def load_books(filename="books.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        console.print(f"Loaded data from {filename}", style="bold green")
        return data
    else:
        console.print(f"{filename} does not exist. Processing authors...", style="bold yellow")
        return None

def generate_embedding(text, task_type="retrieval.query"):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": "jina-embeddings-v3",
        "input": [text],
        "embedding_type": "float",
        "task": task_type
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return embedding
    except requests.exceptions.RequestException as e:
        console.print(f"Failed to generate embedding for text: {text[:30]}... - Error: {e}", style="bold red")
        return None

def get_latest_books_with_embeddings(author, max_results=10):
    encoded_author = quote(author)
    url = f"https://www.googleapis.com/books/v1/volumes?q=inauthor:{encoded_author}&langRestrict=en&maxResults={max_results}&printType=books&orderBy=newest"

    response = requests.get(url)
    response.raise_for_status()
    books = response.json().get("items", [])

    latest_books = []
    for book in books:
        info = book.get("volumeInfo", {})
        description = info.get("description", "N/A")
        
        embedding = generate_embedding(description, task_type="retrieval.passage") if description != "N/A" else None
        
        latest_books.append({
            "title": info.get("title", "N/A"),
            "description": description,
            "publication_date": info.get("publishedDate", "N/A"),
            "thumbnail_url": info.get("imageLinks", {}).get("thumbnail", "N/A"),
            "embedding": embedding
        })

    return latest_books

def rerank_results(query, results, top_n=5):
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": [result["description"] for result in results],
        "top_n": top_n,
        "return_documents": True
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        ranked_results = response.json()["results"]
        reranked = [results[item["index"]] for item in ranked_results]
        for item, rank_info in zip(reranked, ranked_results):
            item["relevance_score"] = rank_info["relevance_score"]
        return reranked
    except requests.exceptions.RequestException as e:
        console.print(f"Failed to rerank results - Error: {e}", style="bold red")
        return results[:top_n]  # Return top N results as fallback

def search_embeddings(search_text, filename="books.json", top_n=5, initial_matches=10):
    data = load_books(filename)
    search_embedding = generate_embedding(search_text, task_type="retrieval.query")
    if search_embedding is None:
        console.print("Failed to generate embedding for the search query.", style="bold red")
        return []

    search_embedding = np.array(search_embedding).reshape(1, -1)  # Reshape for cosine similarity

    matches = []
    for author, books in data.items():
        for book in books:
            book_embedding = book.get("embedding")
            if book_embedding:
                book_embedding = np.array(book_embedding).reshape(1, -1)
                similarity = cosine_similarity(search_embedding, book_embedding)[0][0]
                
                matches.append({
                    "author": author,
                    "title": book.get("title", "N/A"),
                    "description": book.get("description", "N/A"),
                    "publication_date": book.get("publication_date", "N/A"),
                    "thumbnail_url": book.get("thumbnail_url", "N/A"),
                    "similarity": similarity
                })

    # Sort by similarity to get the top `initial_matches`, then rerank for the top 5
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:initial_matches]
    return rerank_results(search_text, matches, top_n)

def save_to_json(data, filename="books.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    console.print(f"Data saved to {filename}", style="bold green")

# Example usage
if __name__ == "__main__":
    books_data = load_books("books.json")
    
    if books_data is None:
        authors = load_authors("authors.txt")
        books_data = {author: get_latest_books_with_embeddings(author) for author in track(authors, description="Processing authors")}
        save_to_json(books_data)

    while True:
        search_text = input("Enter a search term (or type 'exit' to quit): ")
        if search_text.lower() == 'exit':
            console.print("Exiting search...", style="bold green")
            break
        
        results = search_embeddings(search_text, top_n=5, initial_matches=10)

        if results:
            console.print(f"Found {len(results)} matching books:", style="bold green")
            for book in results:
                console.print(f"\nAuthor: {book['author']}")
                console.print(f"Title: {book['title']}")
                console.print(f"Description: {book['description']}")
                console.print(f"Publication Date: {book['publication_date']}")
                console.print(f"Relevance Score: {book.get('relevance_score', 'N/A'):.4f}")
                console.print(f"Thumbnail URL: {book['thumbnail_url']}")
        else:
            console.print("No matching books found.", style="bold red")
