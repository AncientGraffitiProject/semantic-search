import pandas as pd
import numpy as np
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from sentence_transformers import SentenceTransformer

CSV_PATH = "inscriptions.csv"

# 2) Chroma collection name where we'll store XLM-RoBERTa embeddings:
COLLECTION_NAME = "agp_inscriptions_lat_xlm"

# 3) Sentence-Transformers model (multilingual XLM-R variant):
MODEL_NAME = "paraphrase-xlm-r-multilingual-v1"

# 4) Batch sizes:
EMBED_BATCH_SIZE  = 32   # how many inscriptions to embed at once
CHROMA_BATCH_SIZE = 128  # how many vectors to send to Chroma at once


def main():
    # 1. Load CSV
    print(f"Loading CSV from '{CSV_PATH}' ...")
    df = pd.read_csv(CSV_PATH, dtype=str)
    # Clean the data:
    df["latin_text"] = df["latin_text"].fillna("").astype(str)
    total = len(df)
    print(f"Found {total} inscriptions.\n")

    # 2. Load the XLM-RoBERTa SentenceTransformer
    print("Initializing XLM-RoBERTa model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.\n")

    # 3. Iterate in batches, generate embeddings
    all_ids       = []
    all_embeddings = []
    all_metadatas = []

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch_df = df.iloc[i : i + EMBED_BATCH_SIZE]
        texts = batch_df["latin_text"].tolist()
        ids   = batch_df["agpID"].tolist()

        # Encode batch: returns a numpy array 
        embeddings_batch = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True   
        )

        # Convert to Python list-of-lists and collect metadata
        for idx_in_batch, agp_id in enumerate(ids):
            emb = embeddings_batch[idx_in_batch] 
            all_ids.append(str(agp_id))
            all_embeddings.append(emb.tolist())
            all_metadatas.append({
                "agpID": str(agp_id),
                "latin_text": batch_df.iloc[idx_in_batch]["latin_text"]
            })


    # 4. Initialize Chroma client & create/get collection
    print("\nInitializing Chroma client...")
    client = chromadb.PersistentClient(path="db_chroma",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 5. Ingest embeddings into Chroma in batches
    print(f"\nIngesting embeddings into Chroma...")
    for j in range(0, total, CHROMA_BATCH_SIZE):
        sub_ids      = all_ids[j : j + CHROMA_BATCH_SIZE]
        sub_embs     = all_embeddings[j : j + CHROMA_BATCH_SIZE]
        sub_metadata = all_metadatas[j : j + CHROMA_BATCH_SIZE]

        collection.add(
            ids=sub_ids,
            embeddings=sub_embs,
            metadatas=sub_metadata
        )

    # 6. Final confirmation
    print(f"\nDONE. Chroma now holds {collection.count()} vectors.")
    

if __name__ == "__main__":
    main()
