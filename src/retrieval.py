"""
BM25 Retrieval and Natural Retrieval Benchmark Module
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Adheres strictly to Phase 2 Methodology:
- BM25Okapi with k1=1.5, b=0.75, epsilon=0.25.
- Controlled background corpus: 800 training abstracts ONLY.
- Natural retrieval benchmark index: 1,000 abstracts to calculate Recall@3 on 200 test queries.
- Zero-leakage enforcement.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.config import ExperimentConfig


# Standard English stopwords (NLTK-compatible)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}


class BM25Retriever:
    """
    High-performance BM25Okapi retriever implementation adhering to Phase 2 specifications.
    Uses k1=1.5, b=0.75, epsilon=0.25.
    """
    def __init__(self, corpus_records: List[Dict[str, Any]], config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.k1 = self.config.bm25_k1
        self.b = self.config.bm25_b
        self.epsilon = self.config.bm25_epsilon
        self.corpus_records = corpus_records
        
        self.doc_ids = [r["pubid"] for r in corpus_records]
        self.doc_texts = ["\n\n".join(r["context"].get("contexts", [])).strip() for r in corpus_records]
        
        # Tokenize corpus
        self.tokenized_corpus = [self.tokenize(text) for text in self.doc_texts]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.corpus_size = len(self.tokenized_corpus)
        
        # Calculate Document Frequencies (DF)
        self.df = Counter()
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            self.df.update(unique_terms)
            
        # Precompute IDF
        self.idf = {}
        negative_idfs = []
        for term, freq in self.df.items():
            # Standard BM25 IDF formula
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf_val
            if idf_val < 0:
                negative_idfs.append(term)
                
        # Epsilon fallback for negative or near-zero IDF
        eps_val = self.epsilon * (sum(self.idf.values()) / max(1, len(self.idf)))
        for term in negative_idfs:
            self.idf[term] = max(eps_val, 1e-4)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text using r'\\b[a-zA-Z]{2,}\\b', lowercased, minus stopwords."""
        words = re.findall(self.config.token_pattern, text.lower())
        return [w for w in words if w not in STOPWORDS]

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Computes BM25 score between query tokens and document at doc_idx."""
        score = 0.0
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = Counter(doc_tokens)
        
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
        
        for q in query_tokens:
            if q not in term_freqs:
                continue
            tf = term_freqs[q]
            idf = self.idf.get(q, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * len_norm
            score += idf * (numerator / denominator)
        return score

    def retrieve(self, query: str, top_k: int = 3, exclude_pubids: List[str] = None) -> List[Tuple[str, str, float]]:
        """
        Retrieves top_k passages. Returns list of (pubid, text, score).
        Optionally excludes specific pubids.
        """
        exclude_set = set(exclude_pubids or [])
        q_tokens = self.tokenize(query)
        
        scored_docs = []
        for idx in range(self.corpus_size):
            pid = self.doc_ids[idx]
            if pid in exclude_set:
                continue
            s = self.score(q_tokens, idx)
            scored_docs.append((pid, self.doc_texts[idx], s))
            
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        return scored_docs[:top_k]

    def retrieve_disjoint(self, query: str, top_k: int = 3, exclude_pubids: List[str] = None) -> List[Tuple[str, str, float]]:
        """
        Retrieves passages that have zero query keyword overlap (for C2/C3 background/irrelevant passages).
        """
        exclude_set = set(exclude_pubids or [])
        q_tokens = set(self.tokenize(query))
        
        candidates = []
        for idx in range(self.corpus_size):
            pid = self.doc_ids[idx]
            if pid in exclude_set:
                continue
            doc_tokens = set(self.tokenized_corpus[idx])
            overlap = q_tokens.intersection(doc_tokens)
            if len(overlap) == 0:
                candidates.append((pid, self.doc_texts[idx], 0.0))
                if len(candidates) >= top_k:
                    break
                    
        # Fallback if strict zero overlap is too sparse: pick lowest scoring non-zero BM25 docs
        if len(candidates) < top_k:
            all_scored = [(self.doc_ids[i], self.doc_texts[i], self.score(list(q_tokens), i)) 
                          for i in range(self.corpus_size) if self.doc_ids[i] not in exclude_set]
            all_scored.sort(key=lambda x: x[2])
            candidates = all_scored[:top_k]
            
        return candidates[:top_k]


def evaluate_natural_retrieval_recall_at_3(all_records: List[Dict[str, Any]], test_records: List[Dict[str, Any]]) -> float:
    """
    Computes Recall@3 on the natural-retrieval benchmark index containing all 1,000 abstracts.
    Recall@3 = proportion of test queries whose own intact abstract is in top 3 retrieved results.
    """
    print("[Retrieval] Indexing all 1,000 abstracts for natural retrieval benchmark...")
    retriever = BM25Retriever(all_records)
    
    hits = 0
    total = len(test_records)
    for row in test_records:
        target_pubid = row["pubid"]
        question = row["question"]
        results = retriever.retrieve(question, top_k=3)
        retrieved_pubids = [r[0] for r in results]
        if target_pubid in retrieved_pubids:
            hits += 1
            
    recall_at_3 = hits / max(1, total)
    print(f"[Retrieval] Natural Retrieval Benchmark Recall@3: {recall_at_3:.4f} ({hits}/{total})")
    return recall_at_3

