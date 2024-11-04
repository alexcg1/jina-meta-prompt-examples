import os
import sys
from collections import Counter
import argparse

def get_top_authors(library_path, top_n=20):
    """
    Scan the Calibre library and return the top N authors by book count.
    """
    author_counts = Counter()

    # Walk through the library to count books by author
    for root, dirs, files in os.walk(library_path):
        if root == library_path:
            # Each author has their own folder within the Calibre library
            for author_folder in dirs:
                author_path = os.path.join(root, author_folder)
                # Count the number of books (subdirectories) in each author's folder
                book_count = len([name for name in os.listdir(author_path) if os.path.isdir(os.path.join(author_path, name))])
                if book_count > 0:
                    author_counts[author_folder] += book_count
            break  # Only look at the top-level author folders

    # Get the top N authors
    top_authors = author_counts.most_common(top_n)
    return [author for author, count in top_authors]

def save_authors_to_file(authors, filename="authors.txt"):
    """
    Save the list of authors to a file.
    """
    with open(filename, "w") as f:
        for author in authors:
            f.write(f"{author}\n")

def main():
    parser = argparse.ArgumentParser(description="Find top authors in a Calibre library.")
    parser.add_argument("library_path", type=str, help="Path to the Calibre library")
    args = parser.parse_args()

    # Get top authors from the specified Calibre library path
    top_authors = get_top_authors(args.library_path)
    save_authors_to_file(top_authors)
    print(f"Top authors saved to 'authors.txt'.")

if __name__ == "__main__":
    main()
