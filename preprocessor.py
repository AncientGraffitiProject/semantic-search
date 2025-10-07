import re
import unicodedata
from typing import List, Dict, Set
import pandas as pd

class LatinTextPreprocessor:
    """
    Professional-grade Latin text preprocessor for better embeddings.
    Handles normalization, standardization, and semantic enhancement.
    """
    
    def __init__(self):
        # Common Latin abbreviations and their expansions
        self.abbreviations = {
            'D': 'DIS',
            'M': 'MANIBUS',
            'S': 'SACRUM',
            'H': 'HIC',
            'S': 'SITUS',
            'E': 'EST',
            'F': 'FILIUS',
            'L': 'LIBERTUS',
            'C': 'GAIUS',
            'CN': 'GNAEUS',
            'P': 'PUBLIUS',
            'Q': 'QUINTUS',
            'T': 'TITUS',
            'TI': 'TIBERIUS',
            'M': 'MARCUS',
            'L': 'LUCIUS',
            'A': 'AULUS',
            'SP': 'SPURIUS',
            'SER': 'SERVIUS',
            'D': 'DECIMUS',
            'COS': 'CONSUL',
            'IMP': 'IMPERATOR',
            'CAES': 'CAESAR',
            'AVG': 'AUGUSTUS',
            'PONT': 'PONTIFEX',
            'MAX': 'MAXIMUS',
            'TRIB': 'TRIBUNICIA',
            'POT': 'POTESTAS',
            'PROC': 'PROCONSUL',
            'LEG': 'LEGATUS',
            'PR': 'PRAETOR',
            'AED': 'AEDILIS',
            'QUAEST': 'QUAESTOR'
        }
        
        # Common Latin and Greek words for semantic context
        self.semantic_markers = {
            'death': [
                'OBIIT', 'MORTUUS', 'DEFUNCTUS', 'VITA', 'EXCESSIT',
                'ΤΕΤΕΛΕΥΤΗΚΕΝ', 'ΑΠΕΒΙΩΣΕΝ', 'ΘΑΝΑΤΟΣ', 'ΒΙΟΣ', 'ΕΞΕΛΙΠΕΝ'
            ],
            'dedication': [
                'DEDICAVIT', 'POSUIT', 'FECIT', 'FACIUNDUM',
                'ΑΝΕΘΗΚΕΝ', 'ΕΠΟΙΗΣΕΝ', 'ΑΝΕΓΡΑΨΕΝ', 'ΑΝΕΣΤΗΣΕΝ'
            ],
            'family': [
                'FILIUS', 'FILIA', 'PATER', 'MATER', 'UXOR', 'MARITUS',
                'ΥΙΟΣ', 'ΘΥΓΑΤΗΡ', 'ΠΑΤΗΡ', 'ΓΥΝΗ'
            ],
            'religious': [
                'SACRUM', 'DIS', 'DEUS', 'DIVUS', 'TEMPLUM',
                'ΘΕΟΣ', 'ΔΙΟΣ', 'ΝΑΟΣ', 'ΑΓΙΟΣ'
            ],
            'official': [
                'CONSUL', 'PRAETOR', 'TRIBUNUS', 'LEGATUS', 'PROCONSUL',
                'ΥΠΑΤΟΣ', 'ΣΤΡΑΤΗΓΟΣ', 'ΤΡΙΒΟΥΝΟΣ', 'ΛΕΓΑΤΟΣ', 'ΑΝΘΥΠΑΤΟΣ'
            ],
            'memorial': [
                'MONUMENTUM', 'SEPULCRUM', 'TUMULUS', 'MEMORIA',
                'ΤΑΦΟΣ', 'ΤΥΜΒΟΣ'
            ]
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Latin text for better embedding."""
        if not text or pd.isna(text):
            return ""
            
        # Convert to uppercase (standard for Latin inscriptions)
        text = str(text).upper().strip()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove diacritics and non-Latin characters
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,;:')
        
        # Standardize common character variations
        text = text.replace('V', 'U')  # Classical Latin used U for V
        text = text.replace('J', 'I')  # Classical Latin used I for J
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common Latin abbreviations."""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def add_semantic_context(self, text: str) -> str:
        """Add semantic context markers for better embeddings."""
        enhanced_text = text
        
        for category, markers in self.semantic_markers.items():
            for marker in markers:
                if marker in text:
                    enhanced_text += f" [SEMANTIC_{category.upper()}]"
                    break
        
        return enhanced_text
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline."""
        text = self.normalize_text(text)
        text = self.expand_abbreviations(text)
        text = self.add_semantic_context(text)
        return text
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess search query with query expansion."""
        processed = self.preprocess(query)
        
        # Add query expansion for common search terms
        query_expansions = {
            'DEATH': 'DEATH OBIIT MORTUUS DEFUNCTUS ΤΕΤΕΛΕΥΤΗΚΕΝ ΑΠΕΒΙΩΣΕΝ ΘΑΝΑΤΟΣ',
            'FAMILY': 'FAMILY FILIUS FILIA PATER MATER ΥΙΟΣ ΘΥΓΑΤΗΡ ΠΑΤΗΡ',
            'EMPEROR': 'EMPEROR CAESAR AUGUSTUS IMPERATOR ΚΑΙΣΑΡ ΑΥΓΟΥΣΤΟΣ ΑΥΤΟΚΡΑΤΩΡ',
            'DEDICATION': 'DEDICATION POSUIT FECIT DEDICAVIT ΑΝΕΘΗΚΕΝ ΕΠΟΙΗΣΕΝ'
        }
        
        for key, expansion in query_expansions.items():
            if key in processed.upper():
                processed += f" {expansion}"
        
        return processed