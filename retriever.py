import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from langchain_ollama import OllamaEmbeddings # Pakai library baru biar gak warning

class FeedbackRetriever:
    def __init__(self, db_config, ollama_config):
        self.db_config = db_config
        self.embeddings = OllamaEmbeddings(**ollama_config)

    def get_relevant_feedback(self, query_text, limit=5):
        query_vec = self.embeddings.embed_query(query_text)
        
        conn = psycopg2.connect(**self.db_config)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                # Mencari feedback paling mirip menggunakan Cosine Distance (<=>)
                cur.execute("""
                    SELECT komentar FROM user_feedback 
                    ORDER BY embedding <=> %s LIMIT %s
                """, (np.array(query_vec), limit))
                
                results = [row[0] for row in cur.fetchall()]
                return "\n- " + "\n- ".join(results) if results else "No specific feedback found."
        finally:
            conn.close()