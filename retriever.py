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
                # Menggunakan CTE (WITH) agar kita bisa menghitung similarity sekali saja
                # lalu menggunakan hasilnya untuk logika IF/ELSE (CASE) di bawahnya.
                query = """
                    WITH calculated_sim AS (
                        SELECT 
                            sentimen, 
                            komentar, 
                            1 - (embedding <=> %s) AS similarity
                        FROM user_feedback
                    )
                    SELECT sentimen, komentar, similarity
                    FROM calculated_sim
                    ORDER BY 
                        CASE 
                            -- Kasta 1: Negatif/Netral & Lolos Threshold
                            WHEN sentimen IN ('Negatif', 'Netral') AND similarity >= %s THEN 1
                            -- Kasta 2: Positif & Lolos Threshold
                            WHEN sentimen = 'Positif' AND similarity >= %s THEN 2
                            -- Kasta 3: Tidak lolos threshold (diambil kalau slot masih sisa)
                            ELSE 3 
                        END ASC,
                        -- Dalam kasta yang sama, urutkan dari yang paling mirip
                        similarity DESC
                    LIMIT %s;
                """
                
                # Masukkan parameter (query_vec, threshold, threshold, limit)
                cur.execute(query, (np.array(query_vec), similarity_threshold, similarity_threshold, limit))
                
                rows = cur.fetchall()
                
                # Format output agar menampilkan skor similarity di terminal
                results = []
                for row in rows:
                    sentimen = row[0]
                    komentar = row[1]
                    skor = round(row[2], 3) # Bulatkan skor ke 3 desimal
                    
                    # Tampilan: [Negatif | Skor: 0.852] ini komentarnya...
                    results.append(f"[{sentimen} | Skor: {skor}] {komentar}")
                
                return "\n- " + "\n- ".join(results) if results else "Tidak ada feedback spesifik."
                
        except Exception as e:
            logging.error(f"Error retrieving feedback: {e}")
            return "Gagal menarik data dari database."
        finally:
            if 'conn' in locals() and conn:
                conn.close()