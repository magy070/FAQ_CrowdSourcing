import json
from sentence_transformers import SentenceTransformer, util

# Load FAQ data
with open("clean_faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs")

# Load model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# Extract questions and generate embeddings
questions = [faq["question"] for faq in faqs]
print("Generating embeddings...")
question_embeddings = model.encode(questions, convert_to_tensor=True)
print(f"Done! {len(questions)} questions embedded.")

def search(user_query, top_k=3):
    # Encode user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Compute similarity scores
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    
    # Get top results
    top_results = scores.argsort(descending=True)[:top_k]
    
    best_score = scores[top_results[0]].item()

    # Level 1 — exact match
    if best_score > 0.9:
        print("\n Found an exact answer!\n")
        idx = top_results[0].item()
        print(f"Q : {faqs[idx]['question']}")
        print(f"A : {faqs[idx]['answer']}")

    # Level 2 — similar results
    elif best_score > 0.5:
        print("\n🔍 Here are the most relevant FAQs:\n")
        for rank in top_results:
            idx = rank.item()
            score = round(scores[idx].item(), 3)
            print(f"Score : {score}")
            print(f"Q     : {faqs[idx]['question']}")
            print(f"A     : {faqs[idx]['answer']}")
            print("-" * 50)

    # Level 3 — nothing found
    else:
        print("\n Sorry, no relevant FAQ found for your question.")
        print("You can raise this as a new query!")

# Main loop
print("\nFAQ Search Engine Ready!")
print("Type 'exit' to quit\n")

while True:
    query = input("Enter your question: ")
    
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    search(query)
    print()