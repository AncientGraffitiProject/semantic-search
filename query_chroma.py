import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from googletrans import Translator

# Some good query tests: III, roma, ABC, arma, victoria, 

COLLECTION_NAME = "agp_inscriptions_lat_xlm"

# The same model we used for embedding
MODEL_NAME = "paraphrase-xlm-r-multilingual-v1"

def main():

    q_text = input("Enter Latin query: ").strip()

    # 1. Initialize model & embed the query
    print(f"Embedding your query: \"{q_text}\" ...")
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(
        [q_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # 2. Connect to Chroma and fetch collection
    print(f"\nConnecting to Chroma collection '{COLLECTION_NAME}' ...")
    client = chromadb.PersistentClient(
    path="db_chroma",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    # 3. Query for nearest neighbors (set n_results to something reasonably large)
    print(f"Running semantic search (top_k={20}) ...")
    results = collection.query(
    query_embeddings=[q_emb.tolist()],
    n_results=20  # get more so we can filter for <0.75
    )

    ids       = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Initialize translator
    translator = Translator()

    print("\n=== Results Within Distance < 0.75 ===")
    found_any = False
    for rank, (agp_id, md, dist) in enumerate(zip(ids, metadatas, distances), start=1):
        if dist < 0.75:
            found_any = True
            latin_text = md['latin_text']
            # Translate
            try:
                translation = translator.translate(latin_text, src='la', dest='en').text
            except Exception as e:
                translation = f"[Translation Error: {e}]"
            print(f"{rank}. agpID: {agp_id}")
            print(f"    Text: {latin_text}")
            print(f"    Distance: {dist:.4f}")
            print(f"    English: {translation}")
            print("----------------------------")

    if not found_any:
        print("No results found with distance < 0.75.")

if __name__ == "__main__":
    main()