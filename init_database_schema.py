import os
import logging
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# --- IMPORT AI & SCRAPER MODULES ---
from transformers import pipeline
from langchain_ollama import OllamaEmbeddings
from scraper import get_wondr_reviews

# 1. INIT ENVIRONMENT & LOGGING
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 2. KONFIGURASI GLOBAL
DB_CONFIG = {
    "dbname": "prd_db",
    "user": "postgres",
    "password": "postgres",
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

OLLAMA_CONFIG = {
    "model": "qwen3-embedding:4b",
    "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434")
}

VECTOR_DIMENSIONS = 2560

# ==========================================
# DATABASE LAYER
# ==========================================
def get_db_connection():
    """Membuat koneksi ke PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        logging.error(f"Gagal terkoneksi ke database: {e}")
        raise

def init_database_schema(conn):
    """Menyiapkan pgvector dan tabel user_feedback."""
    try:
        with conn.cursor() as cur:
            logging.info("Menyiapkan ekstensi pgvector...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
            
            logging.info("Drop tabel lama dan membuat skema baru dengan kolom 'sentimen'...")
            cur.execute("DROP TABLE IF EXISTS user_feedback;") 
            
            cur.execute(f"""
                CREATE TABLE user_feedback (
                    id SERIAL PRIMARY KEY,
                    kategori VARCHAR(50),
                    sentimen VARCHAR(20),
                    komentar TEXT,
                    embedding VECTOR({VECTOR_DIMENSIONS})
                );
            """)
    except psycopg2.Error as e:
        logging.error(f"Gagal membuat skema database: {e}")
        raise

# ==========================================
# TRANSFORMATION LAYER (AI PROCESSING)
# ==========================================
def get_sentiment(text, sentiment_pipeline):
    """Mendeteksi sentimen menggunakan IndoBERT."""
    try:
        # Potong teks maks 1500 karakter agar tidak melebihi batas token model BERT
        safe_text = text[:1500] 
        result = sentiment_pipeline(safe_text)[0]
        label = result['label'].lower()
        
        # Standarisasi output
        if label == 'positive': return 'Positif'
        if label == 'negative': return 'Negatif'
        return 'Netral'
        
    except Exception as e:
        logging.warning(f"Error sentimen pada teks '{text[:30]}...': {e}")
        return 'Netral' # Fallback aman jika model gagal membaca teks tertentu

def process_and_seed_data(conn, embedder, sentiment_pipeline, feedbacks):
    """Proses ETL: Ekstrak sentimen, ubah ke vektor, simpan ke DB."""
    try:
        with conn.cursor() as cur:
            logging.info(f"Mulai memproses {len(feedbacks)} data review...")
            
            for idx, (kategori, komentar) in enumerate(feedbacks):
                # 1. Analisis Sentimen (IndoBERT)
                sentimen = get_sentiment(komentar, sentiment_pipeline)
                
                # 2. Embedding Vektor (Ollama)
                vec = embedder.embed_query(komentar)
                
                # 3. Insert ke PostgreSQL
                cur.execute(
                    "INSERT INTO user_feedback (kategori, sentimen, komentar, embedding) VALUES (%s, %s, %s, %s)",
                    (kategori, sentimen, komentar, np.array(vec))
                )
                
                # Log progress setiap 20 baris agar terminal tidak terlalu spam
                if (idx + 1) % 20 == 0:
                    logging.info(f"Progress: {idx + 1}/{len(feedbacks)} ulasan tersimpan...")
                    
            logging.info("✅ Proses seeding data selesai dengan sukses!")
            
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat insert data: {e}")
        raise

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    logging.info("=== Memulai Pipeline Data Setup ===")
    
    # 1. Inisialisasi Model AI
    logging.info("Menghubungkan ke Ollama Embeddings...")
    embeddings = OllamaEmbeddings(**OLLAMA_CONFIG)
    
    logging.info("Memuat model IndoBERT (akan download ~400MB jika ini pertama kali)...")
    sentiment_pipe = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
    
    conn = None
    try:
        # 2. Siapkan Database
        conn = get_db_connection()
        init_database_schema(conn)
        
        # 3. Scraping Data (Extract)
        logging.info("Menarik data review terbaru dari Play Store wondr by BNI...")
        scraped_texts = get_wondr_reviews(count=200) 
        
        if not scraped_texts:
            logging.warning("Scraping gagal atau tidak ada review. Menghentikan proses.")
            return

        # 4. Format Data Mentah (Kategori sementara diset statis)
        raw_data = [("Uncategorized", text) for text in scraped_texts]
        
        # 5. Transform & Load ke Database
        process_and_seed_data(conn, embeddings, sentiment_pipe, raw_data)
        
    except Exception as e:
        logging.error(f"Pipeline berhenti karena error: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Koneksi database ditutup.")

if __name__ == "__main__":
    main()