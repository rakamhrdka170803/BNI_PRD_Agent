import psycopg2
import csv
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DB_CONFIG = {
    "dbname": "prd_db",
    "user": "postgres",
    "password": "postgres",
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def export_to_csv():
    try:
        print("Menghubungkan ke database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Mengambil semua data tanpa kolom vektor agar ringan
        cur.execute("""
            SELECT id, sentimen, kategori, komentar 
            FROM user_feedback 
            ORDER BY sentimen, id;
        """)
        
        rows = cur.fetchall()
        
        if not rows:
            print("Database kosong!")
            return
            
        csv_filename = "Data_Review_Wondr.csv"
        
        # Menulis ke file CSV
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Tulis Header
            writer.writerow(['ID', 'Sentimen', 'Kategori', 'Komentar'])
            
            # Tulis Data
            pos = 0; neg = 0; net = 0
            for row in rows:
                writer.writerow(row)
                
                # Hitung statistik
                if row[1] == 'Positif': pos += 1
                elif row[1] == 'Negatif': neg += 1
                else: net += 1
                
        print(f"\n✅ Berhasil mengekspor {len(rows)} data ke file '{csv_filename}'.")
        print("-" * 30)
        print("📊 STATISTIK SENTIMEN:")
        print(f"Positif : {pos}")
        print(f"Negatif : {neg}")
        print(f"Netral  : {net}")
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Terjadi error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    export_to_csv()