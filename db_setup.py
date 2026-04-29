import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_ollama import OllamaEmbeddings
import numpy as np
import logging

# ==========================================
# 1. KONFIGURASI (Environment & Constants)
# ==========================================
# Idealnya ini diambil dari file .env menggunakan library python-dotenv
DB_CONFIG = {
    "dbname": "prd_db",
    "user": "postgres",
    "password": "postgres", # Sesuaikan jika password admin kamu berbeda
    "host": "localhost",
    "port": "5432"
}

OLLAMA_CONFIG = {
    "model": "qwen3-embedding:4b",
    "base_url": "http://localhost:11434"
}

VECTOR_DIMENSIONS = 2560

# Setup Logging agar rapi
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========================================
# 2. DATA DUMMY
# ==========================================
DUMMY_FEEDBACKS = [
    ("Login", "Aplikasi sering freeze kalau pakai biometrik."),
    ("Login", "Gagal login terus padahal password sudah benar."),
    ("Login", "OTP lewat SMS lama sekali sampainya, sudah lewat batas waktu."),
    ("Login", "Fitur 'Remember Me' tidak berfungsi, harus login ulang terus."),
    ("Login", "Face ID sering tidak mengenali wajah saat di tempat redup."),
    ("Login", "Aplikasi tertutup sendiri (force close) sesaat setelah login."),
    ("Login", "Loading screen saat masuk aplikasi memakan waktu lebih dari 1 menit."),
    ("Login", "Reset password link di email masuk ke folder spam."),
    ("Login", "Muncul error 'Connection Timeout' padahal internet stabil."),
    ("Login", "User interface untuk input PIN terlalu kecil, sering salah pencet."),

    # --- TRANSFER ---
    ("Transfer", "Transfer ke bank lain nyangkut tapi saldo sudah terpotong."),
    ("Transfer", "Daftar transfer baru ribet, harus verifikasi berkali-kali."),
    ("Transfer", "BI-FAST sering tidak tersedia di jam-jam sibuk."),
    ("Transfer", "Nama penerima tidak muncul saat input nomor rekening bank lain."),
    ("Transfer", "Fitur jadwal transfer otomatis sering meleset tanggalnya."),
    ("Transfer", "Resi transfer tidak muncul setelah transaksi berhasil."),
    ("Transfer", "Limit transfer harian terlalu rendah untuk akun bisnis."),
    ("Transfer", "Validasi nomor rekening tujuan sangat lambat."),
    ("Transfer", "Riwayat transfer tidak sinkron dengan saldo keluar."),
    ("Transfer", "Proses input kode bank lain kurang user-friendly, harus scroll jauh."),

    # --- UI/UX & NAVIGATION ---
    ("UI/UX", "Warna tombol bayar kurang jelas, hampir sama dengan background."),
    ("UI/UX", "Menu navigasi di bawah terlalu rapat, sering salah klik."),
    ("UI/UX", "Ukuran font di riwayat transaksi terlalu kecil untuk orang tua."),
    ("UI/UX", "Mode gelap (Dark Mode) membuat beberapa teks tidak terbaca."),
    ("UI/UX", "Banner promosi di halaman depan terlalu besar dan mengganggu."),
    ("UI/UX", "Terlalu banyak klik hanya untuk melihat saldo tabungan."),
    ("UI/UX", "Animasi transisi antar menu terasa berat dan bikin pusing."),
    ("UI/UX", "Tombol 'Back' di HP Android sering malah keluar aplikasi."),
    ("UI/UX", "Icon fitur 'Lainnya' tidak deskriptif, membingungkan user."),
    ("UI/UX", "Layout berantakan saat aplikasi dibuka di layar tablet."),

    # --- TOP UP & BILL PAYMENT ---
    ("Top Up", "Top up e-wallet sering delay masuknya sampai 30 menit."),
    ("Top Up", "Pilihan nominal top up tidak fleksibel, ingin input manual."),
    ("Payment", "Bayar tagihan listrik sering gagal dengan pesan 'Provider Down'."),
    ("Payment", "Menu QRIS susah fokus kalau kamera HP kurang bagus."),
    ("Payment", "Tidak ada notifikasi pengingat untuk tagihan rutin bulanan."),
    ("Payment", "Gagal bayar asuransi, saldo balik tapi limit transaksi berkurang."),
    ("Top Up", "Biaya admin top up pulsa lebih mahal dibanding aplikasi sebelah."),
    ("Payment", "Struk pembayaran PDAM tidak bisa didownload dalam format PDF."),
    ("Top Up", "Nomor tujuan top up tidak bisa diambil langsung dari kontak HP."),
    ("Payment", "Update status tagihan setelah bayar memakan waktu lama."),

    # --- SECURITY & PRIVACY ---
    ("Security", "Tidak ada log aktivitas login untuk memantau akses mencurigakan."),
    ("Security", "Sesi aplikasi terlalu cepat habis, harus login ulang terus."),
    ("Security", "Khawatir karena aplikasi minta izin akses kontak dan lokasi."),
    ("Security", "Pesan konfirmasi hapus akun tersembunyi di menu dalam."),
    ("Security", "Ingin ada fitur hide/masking saldo di halaman depan."),
    ("Security", "Verifikasi dua langkah (2FA) hanya via SMS, ingin via Authenticator."),
    ("Security", "Notifikasi transaksi masuk sering telat, bahaya kalau ada pembobolan."),
    ("Security", "Proses ganti nomor HP di aplikasi sangat berbelit-belit."),
    ("Security", "Muncul pop-up mencurigakan saat buka menu promo."),
    ("Security", "Aplikasi bisa di-screenshot, padahal ini data sensitif."),

    # --- INVESTASI & TABUNGAN ---
    ("Investment", "Grafik portofolio investasi sering tidak update real-time."),
    ("Investment", "Beli reksadana prosesnya lama, bisa sampai 3 hari kerja."),
    ("Investment", "Fitur simulasi bunga deposito kurang akurat."),
    ("Investment", "Menu investasi terpisah-pisah, jadi bingung kelola aset."),
    ("Saving", "Buka rekening tambahan secara online sering gagal verifikasi wajah."),
    ("Saving", "Fitur 'Tabungan Berencana' susah dihentikan atau di-edit."),
    ("Saving", "Suku bunga tidak ditampilkan secara transparan di aplikasi."),
    ("Investment", "Tidak ada filter untuk memilih jenis investasi risiko rendah."),
    ("Investment", "Informasi prospektus produk investasi terlalu panjang dan teknis."),
    ("Saving", "Tujuan menabung (goals) tidak bisa diubah fotonya."),

    # --- CUSTOMER SERVICE & NOTIFICATION ---
    ("Service", "Chatbot customer service jawabannya muter-muter terus."),
    ("Service", "Nomor call center susah dihubungi dari luar negeri."),
    ("Notification", "Notifikasi promo masuk terus tiap jam, sangat mengganggu."),
    ("Notification", "Pesan inbox di aplikasi tidak bisa dihapus sekaligus."),
    ("Service", "Antrean live chat sangat panjang, pernah nunggu 2 jam."),
    ("Notification", "Notifikasi push tidak muncul kalau aplikasi tidak dibuka."),
    ("Service", "Email pengaduan dibalas template saja tanpa solusi."),
    ("Notification", "Suara notifikasi transaksi sama dengan notifikasi biasa."),
    ("Notification", "Informasi maintenance aplikasi mendadak, tidak ada info sebelumnya."),
    ("Service", "Lokasi ATM terdekat di peta aplikasi banyak yang sudah tutup."),

    # --- MUTASI & LAPORAN ---
    ("Report", "Cetak e-statement per bulan sering gagal kirim ke email."),
    ("Report", "Filter tanggal di mutasi rekening maksimal hanya 30 hari."),
    ("Report", "Kategori pengeluaran otomatis sering salah (misal: makan masuk ke hobi)."),
    ("Report", "Mutasi hanya menampilkan angka, tidak ada keterangan merchant."),
    ("Report", "Ingin bisa export mutasi ke format Excel/CSV langsung."),
    ("Report", "Grafik pengeluaran bulanan tidak membantu karena terlalu umum."),
    ("Report", "Saldo total tidak langsung update setelah tarik tunai."),
    ("Report", "Pencarian mutasi berdasarkan kata kunci sering tidak ketemu."),
    ("Report", "Tampilan mutasi sangat membosankan, hanya teks hitam putih."),
    ("Report", "Detail transaksi tidak mencantumkan nomor referensi bank."),

    # --- FITUR LAINNYA & MISC ---
    ("Misc", "Aplikasi terasa panas di HP setelah digunakan 5 menit."),
    ("Misc", "Ukuran file aplikasi (storage) terlalu besar, hampir 1 GB."),
    ("Misc", "Aplikasi sering minta update padahal baru saja diupdate."),
    ("Misc", "Fitur tarik tunai tanpa kartu sering error di mesin ATM."),
    ("Misc", "Tidak bisa pakai aplikasi saat menggunakan VPN."),
    ("Misc", "Bahasa di aplikasi campur-campur antara Inggris dan Indonesia."),
    ("Misc", "Loading gambar di menu promo sangat lambat."),
    ("Misc", "Tidak ada fitur split bill untuk nasabah sesama bank."),
    ("Misc", "Integrasi dengan marketplace sering gagal di tahap verifikasi."),
    ("Misc", "Aplikasi tidak support untuk versi OS Android lama."),
    ("Misc", "Widget saldo di homescreen sering tidak sinkron."),
    ("Misc", "Proses scan kartu debit baru sering tidak terbaca."),
    ("Misc", "Fitur bantuan tidak ada fungsi pencarian kata kunci."),
    ("Misc", "Terlalu banyak pop-up survei kepuasan pelanggan."),
    ("Misc", "Tidak ada opsi untuk mematikan suara tombol."),
    ("Misc", "Aplikasi crash saat mencoba mengganti foto profil."),
    ("Misc", "Fitur pencarian di dalam aplikasi tidak relevan hasilnya."),
    ("Misc", "Konsumsi baterai sangat boros saat aplikasi berjalan di background."),
    ("Misc", "Menu favorit sering terhapus sendiri tanpa sebab."),
    ("Misc", "Tidak ada panduan pengguna untuk fitur-fitur baru."),
]

# ==========================================
# 3. FUNGSI DATABASE & EMBEDDING
# ==========================================
def get_db_connection():
    """Membuat dan mengembalikan koneksi database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        logging.error(f"Gagal terkoneksi ke database: {e}")
        raise

def setup_database(conn):
    """Menyiapkan ekstensi pgvector dan membuat tabel jika belum ada."""
    try:
        with conn.cursor() as cur:
            logging.info("Menyiapkan ekstensi pgvector...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
            
            logging.info(f"Membuat tabel dengan dimensi vektor {VECTOR_DIMENSIONS}...")
            # Menghapus tabel lama jika kamu ingin reset data (opsional, uncomment jika perlu)
            # cur.execute("DROP TABLE IF EXISTS user_feedback;") 
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id SERIAL PRIMARY KEY,
                    kategori VARCHAR(50),
                    komentar TEXT,
                    embedding VECTOR({VECTOR_DIMENSIONS})
                );
            """)
    except psycopg2.Error as e:
        logging.error(f"Gagal menyiapkan database: {e}")
        raise

def seed_data(conn, embeddings_model, feedbacks):
    """Melakukan embedding teks dan memasukkannya ke dalam database."""
    try:
        with conn.cursor() as cur:
            # Cek apakah data sudah ada agar tidak duplikat saat di-run ulang
            cur.execute("SELECT COUNT(*) FROM user_feedback;")
            count = cur.fetchone()[0]
            if count > 0:
                logging.info(f"Database sudah berisi {count} data. Melewati proses seeding.")
                return

            logging.info(f"Mulai melakukan embedding dan menyimpan {len(feedbacks)} data...")
            for kategori, komentar in feedbacks:
                # Proses text to vector
                vec = embeddings_model.embed_query(komentar)
                
                # Insert ke database
                cur.execute(
                    "INSERT INTO user_feedback (kategori, komentar, embedding) VALUES (%s, %s, %s)",
                    (kategori, komentar, np.array(vec))
                )
            logging.info("Proses seeding data selesai dengan sukses!")
            
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat seeding data: {e}")
        raise

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    logging.info("Memulai inisialisasi Database dan Embedder...")
    
    # Init Embedder
    embeddings = OllamaEmbeddings(**OLLAMA_CONFIG)
    
    # Eksekusi Flow
    conn = None
    try:
        conn = get_db_connection()
        setup_database(conn)
        seed_data(conn, embeddings, DUMMY_FEEDBACKS)
    except Exception as e:
        logging.error("Proses dihentikan karena terjadi error.")
    finally:
        if conn:
            conn.close()
            logging.info("Koneksi database ditutup.")

if __name__ == "__main__":
    main()