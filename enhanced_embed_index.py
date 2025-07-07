import pandas as pd
import numpy as np
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
from latin_preprocessor import LatinTextPreprocessor

CSV_PATH = "Combined_inscriptions.csv"  # Path to your CSV file
# CSV_PATH = "1964_inscriptions.csv"
COLLECTION_NAME = "latin_and_greek_inscriptions"

# Enhanced model selection - try multiple models for best results
MODELS = [
    "paraphrase-multilingual-mpnet-base-v2",  # Better for semantic similarity
    #"distiluse-base-multilingual-cased",      # Good balance of speed/quality
    #"paraphrase-xlm-r-multilingual-v1"       # Your current model as fallback
]

# Optimized batch sizes
EMBED_BATCH_SIZE = 32   # Increased for better efficiency
CHROMA_BATCH_SIZE = 128  # Increased for better performance

class EnhancedEmbeddingSystem:
    """
    Embedding system with preprocessing and a better model.
    """
    
    def __init__(self, model_name: str = None):
        self.preprocessor = LatinTextPreprocessor()
        self.logger = logging.getLogger(__name__)
        
        # Select best available model
        self.model_name = model_name or self._select_best_model()
        self.model = SentenceTransformer(self.model_name)
        
        # Configure model for better Latin handling
        self._configure_model()
        
        self.logger.info(f"Initialized with model: {self.model_name}")
    
    def _select_best_model(self) -> str:
        """Select the best model for Latin text."""
        for model_name in MODELS:
            try:
                # Test if model can be loaded
                test_model = SentenceTransformer(model_name)
                self.logger.info(f"Successfully loaded model: {model_name}")
                return model_name
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
        
        # Fallback to default
        return MODELS[-1]
    
    def _configure_model(self):
        """Configure model for optimal Latin text processing."""
        # Set model to evaluation mode so you get embeddings
        self.model.eval()
        
        # It sounds like this helps handle historical/ancient text
        if hasattr(self.model, '_modules'):
            for module in self.model._modules.values():
                if hasattr(module, 'dropout'):
                    module.dropout = 0.0 
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts for embedding."""
        processed = []
        for text in texts:
            processed_text = self.preprocessor.preprocess(text)
            processed.append(processed_text)
        
        return processed
    
    def generate_enhanced_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with preprocessing and normalization.
        """
        # Preprocess texts with the above function
        processed_texts = self.preprocess_texts(texts)
        
        # Generates embeddings
        embeddings = self.model.encode(
            processed_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalization should help
            device=None,  # Auto-detect GPU/CPU but I'm able to run it on CPU
            precision="float32"  # This apparently has a good balance between quality and memory
        )
        
        # Additional post-processing for better clustering
        embeddings = self._enhance_embeddings(embeddings, processed_texts)
        
        return embeddings
    
    def _enhance_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """Apply post-processing to improve embeddings."""
        # Apply text-length normalization to reduce length bias
        lengths = np.array([len(text.split()) for text in texts])
        length_weights = 1.0 / (1.0 + 0.1 * np.log(lengths + 1))  # Logarithmic scaling reccomended by people that have done more vector search than me
        
        # Apply weights to reduce length bias
        weighted_embeddings = embeddings * length_weights.reshape(-1, 1)
        
        # Re-normalize after weighting
        norms = np.linalg.norm(weighted_embeddings, axis=1, keepdims=True)
        normalized_embeddings = weighted_embeddings / (norms + 1e-8)
        
        return normalized_embeddings
    
    def create_enhanced_metadata(self, df: pd.DataFrame, index: int) -> Dict:
        """Create enhanced metadata for better search."""
        row = df.iloc[index]
        
        metadata = {
            "agpID": str(row["agpID"]),
            "latin_text": str(row["latin_text"]),
            "processed_text": self.preprocessor.preprocess(str(row["latin_text"])),
            "word_count": len(str(row["latin_text"]).split()),
            "char_count": len(str(row["latin_text"]))
        }
        
        # Add semantic tags so we can filter by them later
        text_upper = str(row["latin_text"]).upper()
        semantic_tags = []
        
        for category, markers in self.preprocessor.semantic_markers.items():
            for marker in markers:
                if marker in text_upper:
                    semantic_tags.append(category)
                    break
        
        metadata["semantic_tags"] = ",".join(semantic_tags)
        
        return metadata

def main():
    """Enhanced main function with better error handling and logging."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load CSV
        logger.info(f"Loading CSV from '{CSV_PATH}' ...")
        df = pd.read_csv(CSV_PATH, dtype=str)
        df = df.dropna(subset=['latin_text'])  # Remove empty texts, but hopefully there isn't any
        total = len(df)
        logger.info(f"Loaded {total} inscriptions")
        
        logger.info("Initializing enhanced embedding system...")
        embedding_system = EnhancedEmbeddingSystem()
        
        # Process in batches
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        
        logger.info("Generating enhanced embeddings...")
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch_df = df.iloc[i : i + EMBED_BATCH_SIZE]
            texts = batch_df["latin_text"].tolist()
            
            # Generate enhanced embeddings
            embeddings_batch = embedding_system.generate_enhanced_embeddings(texts)
            
            # Generate enhanced metadata for each embedding
            for idx_in_batch, (_, row) in enumerate(batch_df.iterrows()):
                emb = embeddings_batch[idx_in_batch]
                metadata = embedding_system.create_enhanced_metadata(batch_df, idx_in_batch)
                
                all_ids.append(metadata["agpID"])
                all_embeddings.append(emb.tolist())
                all_metadatas.append(metadata)
            
            logger.info(f"Processed batch {i//EMBED_BATCH_SIZE + 1}/{(total-1)//EMBED_BATCH_SIZE + 1}")
        
        # Initialize Chroma with some extra settings
        logger.info("Initializing enhanced Chroma client...")
        client = chromadb.PersistentClient(
            path="db_chroma",
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            ),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Enhanced Latin inscriptions with preprocessing"}
        )
        
        logger.info("Ingesting enhanced embeddings into Chroma...")
        for j in range(0, total, CHROMA_BATCH_SIZE):
            sub_ids = all_ids[j : j + CHROMA_BATCH_SIZE]
            sub_embs = all_embeddings[j : j + CHROMA_BATCH_SIZE]
            sub_metadata = all_metadatas[j : j + CHROMA_BATCH_SIZE]
            
            collection.add(
                ids=sub_ids,
                embeddings=sub_embs,
                metadatas=sub_metadata
            )
            
            logger.info(f"Ingested batch {j//CHROMA_BATCH_SIZE + 1}/{(total-1)//CHROMA_BATCH_SIZE + 1}")
        
        # Final confirmation
        final_count = collection.count()
        logger.info(f"SUCCESS! Enhanced Chroma now holds {final_count} vectors.")
        logger.info(f"Model used: {embedding_system.model_name}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()