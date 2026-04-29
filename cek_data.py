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
    csv_filename = "Data_Review_Wondr.csv"
    try:
        print("Menghubungkan ke database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Mengambil semua data (DITAMBAH KOLOM SOURCE)
        cur.execute("""
            SELECT id, source, sentimen, kategori, komentar 
            FROM user_feedback 
            ORDER BY source, sentimen, id;
        """)
        
        rows = cur.fetchall()
        
        if not rows:
            print("Database kosong!")
            return
            
        # Menulis ke file CSV
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Tulis Header (Ditambah Source)
            writer.writerow(['ID', 'Source', 'Sentimen', 'Kategori', 'Komentar'])
            
            # Tulis Data & Hitung Statistik
            pos = 0; neg = 0; net = 0
            playstore = 0; appstore = 0
            
            for row in rows:
                writer.writerow(row)
                
                # row[1] adalah source, row[2] adalah sentimen
                if row[2] == 'Positif': pos += 1
                elif row[2] == 'Negatif': neg += 1
                else: net += 1
                
                if row[1] == 'Play Store': playstore += 1
                elif row[1] == 'App Store': appstore += 1
                
        print(f"\n✅ Berhasil mengekspor {len(rows)} data ke file '{csv_filename}'.")
        print("-" * 30)
        print("📊 STATISTIK SENTIMEN:")
        print(f"Positif : {pos}")
        print(f"Negatif : {neg}")
        print(f"Netral  : {net}")
        print("-" * 30)
        print("📱 STATISTIK SUMBER APLIKASI:")
        print(f"Play Store : {playstore}")
        print(f"App Store  : {appstore}")
        print("-" * 30)
        
    except PermissionError:
        print(f"\n❌ ERROR: File '{csv_filename}' SEDANG TERBUKA!")
        print("💡 Solusi: Silakan tutup dulu file Excel-nya, lalu jalankan ulang script ini.")
    except Exception as e:
        print(f"\n❌ Terjadi error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    export_to_csv()