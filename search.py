import json
from sentence_transformers import SentenceTransformer

# Step 1: Load the FAQ data
with open("clean_faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs")

# Step 2: Load the AI model
print("Loading model... (first time may take a minute)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# Step 3: Extract just the questions
questions = [faq["question"] for faq in faqs]

# Step 4: Generate embeddings for all questions
print("Generating embeddings...")
embeddings = model.encode(questions)

print(f"Done! Each question is now a list of {len(embeddings[0])} numbers")
print(f"Total embeddings created: {len(embeddings)}")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(query, top_k=3):
    # Step 1: Convert the user's query into an embedding
    query_embedding = model.encode([query])
    
    # Step 2: Compare it with all FAQ embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Step 3: Get top_k most similar questions
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Step 4: Return the results
    results = []
    for i in top_indices:
        results.append({
            "question": faqs[i]["question"],
            "answer": faqs[i]["answer"],
            "category": faqs[i]["category"],
            "score": round(float(similarities[i]), 4)
        })
    
    return results

# Test it!
user_query = input("Enter your question: ")
print(f"\nSearching for: '{user_query}'")
print("-" * 50)

results = search(user_query)

if results[0]["score"] > 0.9:
    # Exact match found
    print("Found an exact answer!\n")
    r = results[0]
    print(f"Q : {r['question']}")
    print(f"A : {r['answer']}")

elif results[0]["score"] > 0.5:
    # Similar results
    print("Here are the most relevant FAQs:\n")
    for r in results:
        if r["score"] > 0.5:
            print(f"Score : {r['score']}")
            print(f"Q     : {r['question']}")
            print(f"A     : {r['answer']}")
            print("-" * 50)

else:
    # No good match
    print("Sorry, no relevant FAQ found for your question.")
    print("You can raise this as a new query!") 