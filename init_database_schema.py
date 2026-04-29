import os
import logging
import argparse
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# --- IMPORT AI & SCRAPER MODULES ---
from transformers import pipeline
from langchain_ollama import OllamaEmbeddings
from scraper import get_wondr_playstore_reviews, get_wondr_appstore_reviews

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- KONFIGURASI ---
DB_CONFIG = {
    "dbname": "prd_db", "user": "postgres", "password": "postgres",
    "host": os.getenv("DB_HOST", "localhost"), "port": os.getenv("DB_PORT", "5432")
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
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        logging.error(f"Gagal terkoneksi ke database: {e}")
        raise

def init_database_schema(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
            
            # 1. BUAT TABEL JIKA BELUM ADA (Langsung dengan kolom source)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(50), 
                    kategori VARCHAR(50),
                    sentimen VARCHAR(20),
                    komentar TEXT,
                    embedding VECTOR({VECTOR_DIMENSIONS})
                );
            """)
            
            # 2. TRIK MIGRASI AMAN: 
            # Jika tabel sudah telanjur ada dari versi lama dan belum punya kolom 'source',
            # perintah ini akan otomatis menambahkannya tanpa menghapus data.
            cur.execute("""
                ALTER TABLE user_feedback 
                ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'Play Store';
            """)
    except psycopg2.Error as e:
        logging.error(f"Gagal menyiapkan skema database: {e}")
        raise

def get_existing_comments(conn):
    """Mengambil semua komentar yang sudah ada di database untuk mencegah duplikasi (O(1) Lookup)."""
    with conn.cursor() as cur:
        cur.execute("SELECT komentar FROM user_feedback;")
        return set(row[0] for row in cur.fetchall())

# ==========================================
# TRANSFORMATION LAYER
# ==========================================
def get_sentiment(text, sentiment_pipeline):
    try:
        safe_text = text[:1500] 
        result = sentiment_pipeline(safe_text)[0]
        label = result['label'].lower()
        if label == 'positive': return 'Positif'
        if label == 'negative': return 'Negatif'
        return 'Netral'
    except Exception:
        return 'Netral'

def process_and_seed_data(conn, embedder, sentiment_pipeline, feedbacks):
    existing_comments = get_existing_comments(conn)
    inserted_count = 0
    skipped_count = 0
    
    try:
        with conn.cursor() as cur:
            for source, kategori, komentar in feedbacks:
                # 1. CEK DUPLIKAT DI MEMORI (Sangat Cepat)
                if komentar in existing_comments:
                    skipped_count += 1
                    continue
                
                # 2. Sentimen & Embedding
                sentimen = get_sentiment(komentar, sentiment_pipeline)
                vec = embedder.embed_query(komentar)
                
                # 3. Insert Data Baru dengan Kolom Source
                cur.execute(
                    "INSERT INTO user_feedback (source, kategori, sentimen, komentar, embedding) VALUES (%s, %s, %s, %s, %s)",
                    (source, kategori, sentimen, komentar, np.array(vec))
                )
                
                # 4. Tambahkan teks yang baru diproses ke dalam Set agar tidak duplikat di loop selanjutnya
                existing_comments.add(komentar) 
                inserted_count += 1
                
                if inserted_count % 10 == 0:
                    logging.info(f"Progress: {inserted_count} ulasan baru tersimpan...")
                    
            logging.info(f"✅ Selesai! Ditambahkan: {inserted_count} data baru. Dilewati (Duplikat): {skipped_count} data.")
            
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat insert data: {e}")
        raise

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Wondr by BNI Review Scraper & Seeder")
    parser.add_argument(
        '--target', 
        type=str, 
        choices=['playstore', 'appstore', 'both'], 
        default='both', 
        help="Pilih target scraping: 'playstore', 'appstore', atau 'both'."
    )
    args = parser.parse_args()

    logging.info(f"=== Menjalankan Pipeline ETL untuk Target: {args.target.upper()} ===")
    
    embeddings = OllamaEmbeddings(**OLLAMA_CONFIG)
    sentiment_pipe = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
    
    conn = None
    try:
        conn = get_db_connection()
        init_database_schema(conn)
        
        raw_data = []
        
        # Eksekusi berdasarkan Argumen CLI (Masing-masing 100 data)
        if args.target in ['playstore', 'both']:
            texts = get_wondr_playstore_reviews(count=100)
            raw_data.extend([("Play Store", "Uncategorized", t) for t in texts])
            
        if args.target in ['appstore', 'both']:
            texts = get_wondr_appstore_reviews(count=100)
            raw_data.extend([("App Store", "Uncategorized", t) for t in texts])
        
        if not raw_data:
            logging.warning("Tidak ada data ulasan yang berhasil ditarik.")
            return

        process_and_seed_data(conn, embeddings, sentiment_pipe, raw_data)
        
    except Exception as e:
        logging.error(f"Pipeline berhenti karena error: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Koneksi database ditutup.")

if __name__ == "__main__":
    main()