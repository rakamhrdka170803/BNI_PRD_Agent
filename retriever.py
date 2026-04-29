import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from langchain_ollama import OllamaEmbeddings
import logging

class FeedbackRetriever:
    def __init__(self, db_config, ollama_config):
        self.db_config = db_config
        self.embeddings = OllamaEmbeddings(**ollama_config)

    def get_relevant_feedback(self, query_text, limit=10, similarity_threshold=0.6):
        """
        Menarik feedback dengan aturan:
        1. Prioritas 1: Negatif/Netral dengan similarity >= threshold
        2. Prioritas 2: Positif dengan similarity >= threshold
        3. Prioritas 3: Sisanya (diurutkan berdasarkan similarity tertinggi)
        """
        try:
            query_vec = self.embeddings.embed_query(query_text)
            conn = psycopg2.connect(**self.db_config)
            register_vector(conn)
            
            with conn.cursor() as cur:
                # ... di dalam fungsi get_relevant_feedback ...
                query = """
                    WITH calculated_sim AS (
                        SELECT 
                            source, -- Tambahkan source
                            sentimen, 
                            komentar, 
                            1 - (embedding <=> %s) AS similarity
                        FROM user_feedback
                    )
                    SELECT source, sentimen, komentar, similarity
                    FROM calculated_sim
                    ORDER BY 
                        CASE 
                            WHEN sentimen IN ('Negatif', 'Netral') AND similarity >= %s THEN 1
                            WHEN sentimen = 'Positif' AND similarity >= %s THEN 2
                            ELSE 3 
                        END ASC,
                        similarity DESC
                    LIMIT %s;
                """
                
                cur.execute(query, (np.array(query_vec), similarity_threshold, similarity_threshold, limit))
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    source = row[0]
                    sentimen = row[1]
                    komentar = row[2]
                    skor = round(row[3], 3)
                    
                    # Tampilan: [App Store] [Negatif | Skor: 0.852] komentar...
                    results.append(f"[{source}] [{sentimen} | Skor: {skor}] {komentar}")
                
                return "\n- " + "\n- ".join(results) if results else "Tidak ada feedback spesifik."  
        except Exception as e:
            logging.error(f"Error retrieving feedback: {e}")
            return "Gagal menarik data dari database."
        finally:
            if 'conn' in locals() and conn:
                conn.close()