print("DEBUG: New version running")

from sentence_transformers import SentenceTransformer, util
import json

# Load FAQ data
with open("clean_faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract questions
questions = [faq["question"] for faq in faqs]

# Create embeddings
question_embeddings = model.encode(
    questions,
    convert_to_tensor=True
)

def semantic_search(user_query, top_k=3):
    query_embedding = model.encode(
        user_query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        question_embeddings
    )[0]

    best_score = scores.max().item()
    print("Best score:", round(best_score, 3))

    if best_score < 0.65:
        print("Sorry, I couldn't find a confident answer.")
        return

    top_results = scores.argsort(descending=True)[:top_k]

    print("\nTop Matches:\n")

    for idx in top_results:
        idx = idx.item()  # convert tensor index to integer

        print("Question:", faqs[idx]["question"])
        print("Answer:", faqs[idx]["answer"])
        print("Score:", round(scores[idx].item(), 3))
        print("-" * 50)

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    semantic_search(query)