!pip install -q datasets

from datasets import load_dataset

# Streams public-domain books; filter for German ("de")
dataset = load_dataset("manu/project_gutenberg", split="de", streaming=True)

with open("books_set.txt", "w", encoding="utf-8") as f:
    for i, book in enumerate(dataset):
        f.write(book["text"] + "\n\n")
        if i >= 300:  # First 300 full German books (~20-30 million words!)
            break

print("Saved 300 books to books_set.txt!")
