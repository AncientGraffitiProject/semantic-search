import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import re
import logging
from typing import List, Dict, Set
from latin_preprocessor import LatinTextPreprocessor
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from tabulate import tabulate
import textwrap
import sys
import io


import translator_test

# Very good ones to look at: Obiit, Victoria, amor, αγάπη, lover, amat, medicamentum, donum, urbe, in forma navis scriptum, city, stupid
# Good ones: Κίνναμοϲ, piscator, scripsit, carmina, medicine, happy, fool, hominis/vale, numini

COLLECTION_NAME = "latin_and_greek_inscriptions"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # Updated to better model
DB_CHROMA_PATH = "db_chroma"

# Load environment variables
load_dotenv()

# Enhanced search parameters
SEMANTIC_DISTANCE_THRESHOLD = 0.5  # Kinda strict threshold for better results
KEYWORD_BOOST_FACTOR = 0.2  # If the keyword is in the inscription, boost the score
TOP_K_INITIAL = 100  # Get more candidates for re-ranking
TOP_K_FINAL = 25   # Final results to show

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
        self._initialize_gemini()
        
    
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
    
    def _initialize_gemini(self):
        """Initialize Gemini LLM."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                self.gemini_available = True
                print("Gemini initialized for reranking")
            else:
                self.gemini_available = False
                print("No GEMINI_API_KEY found - no LLM reranking")
        except Exception as e:
            self.gemini_available = False
            print(f"Gemini initialization failed: {e} - no LLM reranking")

    def _llm_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Use Gemini to rerank search results based on relevance."""
        # Skips the rest of this function if there is nothing to do.
        if not self.gemini_available or len(results) == 0:
            return results
        
        try:
            # Prepare data for LLM
            results_for_llm = []
            for i, (agp_id, result_data) in enumerate(results):
                metadata = result_data['metadata']
                results_for_llm.append({
                    'index': i,
                    'agp_id': agp_id,
                    'latin_text': metadata.get('latin_text', ''),
                    'semantic_score': result_data.get('semantic_score', 0),
                    'keyword_score': result_data.get('keyword_score', 0),
                    'match_type': result_data.get('match_type', '')
                })
            
            # Create prompt for Gemini
            prompt = f"""You are an expert in Ancient Greek and Latin, epigraphy, and ancient inscriptions. 
Analyze these search results for the query: "{query}"

Your task is to filter and rerank the results. You should:
1. Exclude any inscriptions that are completely unrelated to the query concept or theme
2. Keep inscriptions with direct word matches, semantic connections, or thematic relevance
3. Rank the remaining relevant results by their importance to the query

For example:
- If query is "amat" (love), Keep: amor, Venus, marriage, flowers, beauty, passion
- If query is "amat" (love), Exclude: communem (common), random names with no love context
- If query is "victoria" (victory), Keep: triumph, conquest, military terms, success
- If query is "victoria" (victory), Exclude: unrelated personal names or everyday activities

Make sure not to prioritize results based off the language that the query is in, but rather the meaning of all the words and their relevance to the query.

Evaluation criteria (in order of importance):
1. Direct semantic relationship to the query concept
2. Thematic or cultural connection (e.g., Venus relates to love, laurel relates to victory)
3. Historical context relevance
4. Exact word matches (but only if contextually appropriate)

Results to evaluate:
"""
            # Add each inscription to the prompt along with the scores that I calculated
            for item in results_for_llm:
                prompt += f"\n{item['index']}. ID: {item['agp_id']}\n"
                prompt += f"   Latin or Greek: {item['latin_text']}\n"
                prompt += f"   Scores: Semantic={item['semantic_score']:.3f}, Keyword={item['keyword_score']:.3f}\n"
            
            prompt += f"""
MAKE SURE to only return  the indices of relevant inscriptions as a comma-separated list, ranked by relevance (most relevant first).
Exclude completely unrelated entries. If no results are relevant, return an empty response.
Example format: 2,0,5,1 (only include relevant indices)

Your filtered and ranked indices:"""
            
            # Get Gemini response
            response = self.gemini_model.generate_content(prompt)
            ranking_text = response.text.strip()
            
            # Parse the ranking
            try:
                ranking_text = ranking_text.strip()
                
                # Handle empty response (no relevant results)
                if not ranking_text or ranking_text.lower() in ['none', 'empty', 'no results']:
                    print("LLM filtered out all results as irrelevant")
                    return []
                
                
                # Validate the indeces
                indices = [int(x.strip()) for x in ranking_text.split(',') if x.strip()]
                valid_indices = [i for i in indices if 0 <= i < len(results)]
                
                # Add any missing indices at the end for if there are high-scoring keyword matches that the LLM might have overlooked
                missing_indices = []
                for i, (agp_id, result_data) in enumerate(results):
                    if i not in valid_indices:
                        # Include if it has a very high keyword score
                        if result_data.get('keyword_score', 0) >= 1.5:
                            missing_indices.append(i)
                
                # Add missing high-relevance results at the end
                valid_indices.extend(missing_indices[:3])  # Only 3 additional results so it doesn't undo everthing
                
                # Reorder results based on LLM ranking
                reranked_results = []
                for idx in valid_indices:
                    if idx < len(results):
                        agp_id, result_data = results[idx]
                        result_data['llm_reranked'] = True
                        reranked_results.append((agp_id, result_data))
                
                filtered_count = len(results) - len(reranked_results)
                print(f"LLM reranked {len(reranked_results)} results (filtered out {filtered_count} irrelevant)")
                return reranked_results
                
            except (ValueError, IndexError) as e:
                print(f"Couldn't parse LLM ranking: {e}")
                return results
                
        except Exception as e:
            print(f"LLM reranking failed: {e}")
            return results   
    
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
                'match_type': 'semantic',
                'llm_reranked': False
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
                    'match_type': 'keyword',
                    'llm_reranked': False
                }
        
        # Sort by combined score and only keep the top K results
        sorted_results = sorted(
            combined_results.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        initial_results = sorted_results[:TOP_K_FINAL]
        
        # Apply LLM reranking
        final_results = self._llm_rerank(initial_results, query)
        
        return final_results
    
    def _get_translation(self, latin_text: str) -> str:
        """Get translation for Latin text synchronously."""
        try:
            # Run the async translation function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Redirect stdout to capture the translation
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            loop.run_until_complete(translator_test.translate_text(latin_text))
            translation = captured_output.getvalue().strip()
                    
            sys.stdout = old_stdout
            loop.close()
            
            return translation
            
        except Exception as e:
            return f"Translation error: {str(e)[:30]}..."
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        if not text:
            return ""
        return '\n'.join(textwrap.wrap(text, width=width))
    
    def _format_results_table(self, results: List[tuple]) -> None:
        """Format results as a table."""
        if not results:
            print("No results found.")
            return
        
        # Prepare table data
        table_data = []
        headers = ["Rank", "agpID", "Text", "Match Type", "Semantic Score", "Translation"]
        
        for rank, (agp_id, result_data) in enumerate(results, 1):
            metadata = result_data['metadata']
            latin_text = metadata['latin_text']
            
            # Get translation
            translation = self._get_translation(latin_text)
            
            # Wrap text for better display - adjust width based on content
            text_width = min(30, max(15, len(latin_text) // 2))
            translation_width = min(35, max(20, len(translation) // 2))
            
            wrapped_text = self._wrap_text(latin_text, text_width)
            wrapped_translation = self._wrap_text(translation, translation_width)
            
            # Format semantic score - always show it
            if result_data['semantic_score'] == 0:
                semantic_score = "N/A"
            else:
                semantic_score = f"{result_data['semantic_score']:.4f}"
            
            # Format match type
            match_type = result_data['match_type'].upper()
            
            table_data.append([
                rank,
                agp_id,
                wrapped_text,
                match_type,
                semantic_score,
                wrapped_translation
            ])
        
        # Print table
        print(f"\n=== Top {len(results)} Results ===")
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
        print()  # Add extra line for readability
    
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
        if self.gemini_available:
            print("Applying LLM reranking...")
        final_results = self._rerank_results(semantic_results, keyword_matches, query)
        
        # Display results using table format
        self._format_results_table(final_results)
        
        # Display search statistics
        if final_results:
            semantic_count = len([r for _, r in final_results if r['match_type'] in ['semantic', 'hybrid']])
            keyword_count = len([r for _, r in final_results if r['match_type'] in ['keyword', 'hybrid']])
            hybrid_count = len([r for _, r in final_results if r['match_type'] == 'hybrid'])
            llm_reranked_count = len([r for _, r in final_results if r.get('llm_reranked', False)])
            
            print(f"\nSearch Statistics:")
            print(f"- Semantic matches: {semantic_count}")
            print(f"- Keyword matches: {keyword_count}")
            print(f"- Hybrid matches: {hybrid_count}")
            if self.gemini_available:
                print(f"- LLM reranked results: {llm_reranked_count}")

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