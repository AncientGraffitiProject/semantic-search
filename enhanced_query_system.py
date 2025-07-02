import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import re
import logging
from typing import List, Dict, Set
from latin_preprocessor import LatinTextPreprocessor
import asyncio

import translator_test

# Interesting ones to look at: Obiit, Victoria, amor, αγάπη, medicamentum, donum, urbe, in forma navis scriptum, Quam mita vis Pastores atque Caste monit erga pugiles

# TODO Try only using certain parts of the code to see if it makes a measurable improvement.

#TODO put in Llama 3.2 or a small 4 one to rerank the results.

#COLLECTION_NAME = "latin_agp_inscriptions"
COLLECTION_NAME = "1964_latin_agp_inscriptions"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # Updated to better model
DB_CHROMA_PATH = "db_chroma"

# Enhanced search parameters
SEMANTIC_DISTANCE_THRESHOLD = 0.5  # Kinda strict threshold for better results
KEYWORD_BOOST_FACTOR = 0.2  # If the keyword is in the inscription, boost the score
TOP_K_INITIAL = 100  # Get more candidates for re-ranking
TOP_K_FINAL = 50   # Final results to show

class EnhancedLatinQuerySystem:
    """
    Improved Latin query system with a combination of static and vector search capabilities, and re-ranking.
    """
    
    def __init__(self):
        
        self.preprocessor = LatinTextPreprocessor()

        # Had to include a logger for some of the translation stuff.
        logging.getLogger().setLevel(logging.WARNING)

        
        # Initialize components so you can search multiple times
        self._initialize_model()
        self._initialize_chroma()
        
    
    def _initialize_model(self):
        """Initialize the embedding model with optimal settings."""
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            self.model.eval()  # Set to evaluation mode, so it is able to generate consistent embeddings with context
        except Exception as e:
            self.model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1") # Fallback model
    
    def _initialize_chroma(self):
        """Initialize Chroma client and collection."""
        self.client = chromadb.PersistentClient(
            path=DB_CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
    
    def _enhanced_tokenize(self, text: str) -> Set[str]:
        """Advanced tokenization with Latin-specific features."""
        if not text:
            return set()
        
        # Normalize text
        text = self.preprocessor.normalize_text(text)
        
        # Setup a set to hold all the tokens
        tokens = set()
        
        # Word-level tokens
        words = re.findall(r"\b\w+\b", text.lower())
        tokens.update(words)
        
        # Add bigrams for better phrase matching
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            tokens.add(bigram)
        
        # Add trigrams for names and titles
        for i in range(len(words) - 2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            tokens.add(trigram)
        
        return tokens
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        expanded = self.preprocessor.preprocess_query(query)
        
        # Add domain-specific expansions
        query_upper = query.upper()
        
        # Death-related terms
        if any(term in query_upper for term in ['DEATH', 'DIE', 'DEAD']):
            expanded += " OBIIT MORTUUS DEFUNCTUS VITA EXCESSIT MORIOR"
        
        # Family-related terms
        if any(term in query_upper for term in ['FAMILY', 'SON', 'DAUGHTER', 'FATHER', 'MOTHER']):
            expanded += " FILIUS FILIA PATER MATER UXOR MARITUS"
        
        # Religious terms
        if any(term in query_upper for term in ['GOD', 'SACRED', 'RELIGIOUS']):
            expanded += " DEUS SACRUM DIS DIVUS TEMPLUM"
        
        # Official titles
        if any(term in query_upper for term in ['EMPEROR', 'CONSUL', 'OFFICIAL']):
            expanded += " IMPERATOR CAESAR AUGUSTUS CONSUL PRAETOR"
        
        return expanded
    
    def _semantic_search(self, query: str) -> List[Dict]:
        """Perform enhanced semantic search."""
        # Expand and preprocess query
        expanded_query = self._expand_query(query)
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [expanded_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Search with larger initial results for re-ranking
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K_INITIAL,
            include=['metadatas', 'distances']
        )
        
        return results
    
    def _keyword_search(self, query: str) -> Dict[str, Dict]:
        """Exact search through the processed and unprocessed collection. Also called fuzzy matching."""
        query_tokens = self._enhanced_tokenize(query)
        
        # Get all inscriptions for keyword matching
        all_data = self.collection.get(
            include=["metadatas"], 
            limit=50000
        )
        
        keyword_matches = {}
        
        for idx, metadata in enumerate(all_data["metadatas"]):
            if not metadata or "latin_text" not in metadata:
                continue
                
            # Get the OG and processed text tokens
            text_tokens = self._enhanced_tokenize(metadata["latin_text"])
            processed_tokens = self._enhanced_tokenize(metadata.get("processed_text", ""))
            
            # Calculate how much they match the query
            exact_matches = len(query_tokens & text_tokens)
            processed_matches = len(query_tokens & processed_tokens)
            
            total_matches = exact_matches + (processed_matches * 0.8)  # Processed matches matter a little less
            
            if total_matches > 0:
                agp_id = metadata.get("agpID", f"row_{idx}")
                match_score = total_matches / len(query_tokens)  # Normalize the score
                
                keyword_matches[agp_id] = {
                    'metadata': metadata,
                    'score': match_score,
                    'exact_matches': exact_matches,
                    'processed_matches': processed_matches
                }
        
        return keyword_matches
    
    def _rerank_results(self, semantic_results: List[Dict], keyword_matches: Dict[str, Dict], query: str) -> List[Dict]:
        """Re-rank the results by combining semantic and keyword scores."""
        combined_results = {}
        
        # Process semantic results
        ids = semantic_results["ids"][0]
        metadatas = semantic_results["metadatas"][0]
        distances = semantic_results["distances"][0]
        
        for agp_id, metadata, distance in zip(ids, metadatas, distances):
            # Skip results that don't meet the semantic similarity threshold
            if distance > SEMANTIC_DISTANCE_THRESHOLD:
                continue
                
            semantic_score = max(0, 1 - distance)  # Convert distance to positive similarity score
            
            combined_results[agp_id] = {
                'metadata': metadata,
                'semantic_score': semantic_score,
                'keyword_score': 0,
                'combined_score': semantic_score,
                'distance': distance,
                'match_type': 'semantic'
            }
        
        # Add keyword scores and boost matches
        for agp_id, keyword_data in keyword_matches.items():
            if agp_id in combined_results:
                # Boost existing semantic matches that also have keyword matches
                combined_results[agp_id]['keyword_score'] = keyword_data['score']
                combined_results[agp_id]['combined_score'] = (
                    combined_results[agp_id]['semantic_score'] + 
                    (keyword_data['score'] * KEYWORD_BOOST_FACTOR)
                )
                combined_results[agp_id]['match_type'] = 'hybrid'
            else:
                # Add pure keyword matches
                combined_results[agp_id] = {
                    'metadata': keyword_data['metadata'],
                    'semantic_score': 0,
                    'keyword_score': keyword_data['score'],
                    'combined_score': keyword_data['score'] * 0.8,  # Slightly lower weight for pure keyword since just because it has the same word doesn't mean it is relevant.
                    'distance': 1.0,  # Max distance for keyword-only
                    'match_type': 'keyword'
                }
        
        # Sort by combined score and only keep the top K results
        sorted_results = sorted(
            combined_results.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        return sorted_results[:TOP_K_FINAL]
    
    def _format_result(self, agp_id: str, result_data: Dict, rank: int) -> None:
        """Format the search result with translation."""
        metadata = result_data['metadata']
        latin_text = metadata['latin_text']
        
        # Format output
        print(f"{rank}. agpID: {agp_id}")
        print(f"    Text: {latin_text}")
        
        if result_data['match_type'] != 'keyword':
            print(f"    Semantic Score: {result_data['semantic_score']:.4f}")
        
        if result_data['keyword_score'] > 0:
            print(f"    Keyword Score: {result_data['keyword_score']:.4f}")
        
        print(f"    Match Type: {result_data['match_type'].upper()}")
        print(f"    Translation: ")
        asyncio.run(translator_test.translate_text(latin_text))
        
        # Add semantic tags if available
        if 'semantic_tags' in metadata and metadata['semantic_tags']:
            print(f"    Categories: {metadata['semantic_tags']}")
        
        print("----------------------------")
        
        return
    
    def search(self, query: str) -> None:
        """Main search function with enhanced capabilities."""
        if not query or not query.strip():
            print("No query entered. Please try again.")
            return
        
        print(f"Searching for: \"{query}\"")
        print("=" * 50)
        
        print("Performing semantic and keyword search...")
        semantic_results = self._semantic_search(query)
        keyword_matches = self._keyword_search(query)
        
        print("Re-ranking and combining results...")
        final_results = self._rerank_results(semantic_results, keyword_matches, query)
        
        # Display results
        if not final_results:
            print("No results found.")
            return
        
        print(f"\n=== Top {len(final_results)} Results ===")
        
        for rank, (agp_id, result_data) in enumerate(final_results, 1):
            self._format_result(agp_id, result_data, rank)
        
        # Display search statistics
        semantic_count = len([r for _, r in final_results if r['match_type'] in ['semantic', 'hybrid']])
        keyword_count = len([r for _, r in final_results if r['match_type'] in ['keyword', 'hybrid']])
        hybrid_count = len([r for _, r in final_results if r['match_type'] == 'hybrid'])
        
        print(f"\nSearch Statistics:")
        print(f"- Semantic matches: {semantic_count}")
        print(f"- Keyword matches: {keyword_count}")
        print(f"- Hybrid matches: {hybrid_count}")

def main():
    """Puts everything together with a user-friendly interface."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Initializing Enhanced Latin Query System...")
    print("This may take a moment on first run...")
    
    try:
        query_system = EnhancedLatinQuerySystem()
        print("System ready! You can now search Latin inscriptions.\n")
        
        while True:
            try:
                query = input("Enter Latin query (or 'quit' to exit): ").strip()
                
                # Handle exit commands
                if query.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                # Makes sure the user actually put something
                if not query:
                    print("Please enter a search query.")
                    continue
                
                # Perform search
                query_system.search(query)
                
                # Signify start of search
                print("\n" + "=" * 50)
                
            except KeyboardInterrupt:
                print("\nSearch interrupted. Enter 'quit' to exit.")
                continue
            except Exception as e:
                print(f"Search error: {e}")
                continue
                
    except Exception as e:
        print(f"Failed to initialize system: {e}")

if __name__ == "__main__":
    main()