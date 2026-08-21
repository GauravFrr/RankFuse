import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rankfuse.exceptions import StoreError

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt",
    "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt",
    "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here",
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i",
    "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its",
    "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she",
    "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than",
    "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
    "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", "what",
    "whats", "when", "whens", "where", "wheres", "which", "while", "who",
    "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you",
    "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}


def clean_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 (lowercase, remove punctuation, filter stopwords)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in ENGLISH_STOPWORDS]


class BM25Index:
    """Keyword-based sparse retrieval index using BM25 Okapi."""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.index_path = self.persist_dir / "bm25_index.pkl"
        self.bm25 = None
        self.doc_ids = []

        if self.index_path.exists():
            self._load()

    def build(self, documents: list[dict]) -> None:
        """Build the BM25 index over a list of documents and save it to disk.

        Args:
            documents: A list of dicts, each with 'id' and 'text'.
        """
        if not documents:
            self.bm25 = None
            self.doc_ids = []
            return

        try:
            self.doc_ids = [doc["id"] for doc in documents]
            corpus = [clean_tokenize(doc["text"]) for doc in documents]
            self.bm25 = BM25Okapi(corpus)

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)
        except Exception as e:
            raise StoreError(f"Failed to build BM25 index: {e}")

    def query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Query the BM25 index for keyword matches.

        Args:
            query_text: The search query.
            top_k: Number of results to return.

        Returns:
            A list of tuples, each containing the document ID and its BM25 score.
        """
        if not self.bm25 or not query_text:
            return []

        try:
            tokenized_query = clean_tokenize(query_text)
            if not tokenized_query:
                return []

            scores = self.bm25.get_scores(tokenized_query)
            results = []
            for doc_id, score in zip(self.doc_ids, scores):
                if score > 0.0:
                    results.append((doc_id, float(score)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            raise StoreError(f"Failed to query BM25 index: {e}")

    def _load(self) -> None:
        """Load the BM25 index from disk."""
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.doc_ids = data["doc_ids"]
        except Exception as e:
            raise StoreError(f"Failed to load BM25 index from disk: {e}")
