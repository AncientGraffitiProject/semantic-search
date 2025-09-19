# Epigraphic Semantic Search

## Overview

**Epigraphic Semantic Search** is a digital humanities tool designed to transform the way researchers explore ancient Greek and Latin inscriptions. Traditional epigraphic databases rely mainly on keyword matching, which often misses semantically related texts. This project introduces a hybrid search system that leverages vector embeddings and semantic similarity, enabling users to discover thematically related inscriptions—even when exact keywords do not match.

The system combines:

- **Semantic Search:** Uses high-dimensional vector representations to capture the meanings of inscriptions.
- **Keyword Matching:** Ensures that exact or near-exact matches are not missed.
- **Large Language Model (LLM) Re-ranking:** Applies state-of-the-art LLMs to filter and rank results by thematic relevance.

The pipeline and code are fully reproducible, open-source, and work directly on open data from the Ancient Graffiti Project.

## Features

- **Semantic Embedding:** Inscriptions are embedded using advanced multilingual transformer models, capturing meaning beyond surface text.
- **Custom Preprocessing:** Texts are normalized and enriched for improved Latin and Greek semantic handling.
- **Hybrid Querying:** Combines semantic similarity with traditional keyword matches for comprehensive search results.
- **LLM Reranking:** Integrates a large language model (Gemini) to further refine search results by thematic and contextual relevance.
- **Translation Support:** Automatically translates results for easier interpretation.
- **Batch Processing & Metadata:** Efficient vectorization and rich metadata (word/char counts, semantic tags) for each inscription.
- **Command-Line Interface:** Search interactively from the terminal.

## Data

Inscriptions are sourced from the **Ancient Graffiti Project** and stored in `Combined_inscriptions.csv`.

Each row contains:

- `agpID`
- `latin_text`

## Installation

### Requirements:

- Python 3.11.5

Install all dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage

### Building the Embedding Index

To process and embed the inscriptions, run:

```bash
python enhanced_embed_index.py
```

This will:

- Preprocess and embed all inscriptions.
- Store embeddings and metadata in a **ChromaDB** persistent local database.

### Searching the Corpus

In order to enable LLM reranking, in a .env file add a Google Gemini API key which you can get for free here: https://aistudio.google.com/apikey

To search inscriptions with a query (in Latin, Greek, or English):

```bash
python enhanced_query_system.py
```

Enter your query at the prompt.  
Results will be printed in a formatted table, including translations and match details.

## File Structure

| File | Description |
| --- | --- |
| `enhanced_embed_index.py` | Preprocesses text, generates embeddings, and ingests into ChromaDB. |
| `enhanced_query_system.py` | Hybrid search interface (semantic, keyword, LLM re-ranking); command-line interaction. |
| `latin_preprocessor.py` | Professional-grade normalization and semantic enhancement for Latin/Greek text. |
| `translator_test.py` | Asynchronous translation helper (Latin/Greek to English). |
| `Combined_inscriptions.csv` | The main data file (inscriptions). |
| `requirements.txt` | All necessary dependencies. |

## Reproducibility

- All code and data are **open and documented**.
- The embedding and query pipelines are **deterministic** and easily rerun.
- Environment variables (e.g., `GEMINI_API_KEY` for LLM reranking) are loaded via `.env`.

## Results & Research Impact

This system demonstrates the ability to identify inscriptions that are semantically related, surfacing connections missed by traditional search. It opens new avenues for epigraphic research and interdisciplinary discovery in the digital humanities.

## Acknowledgments

- **Inscriptions data:** Ancient Graffiti Project  
- **Vector DB:** ChromaDB  
- **Embeddings:** Sentence Transformers  
- **LLM reranking:** Gemini API

## License

This project is open-source and licensed under the **MIT License**.

## Contact

For questions or contributions, open an issue or contact **soccerlover29** or **sprenks18**.
