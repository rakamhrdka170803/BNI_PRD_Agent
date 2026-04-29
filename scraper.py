from google_play_scraper import Sort, reviews
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_wondr_reviews(count=100):
    """
    Mengambil review dari aplikasi wondr by BNI di Play Store.
    Menggunakan API, bukan scraping UI, agar anti-patah.
    """
    logging.info(f"Mulai mengambil {count} review dari wondr by BNI...")
    
    try:
        result, _ = reviews(
            'id.bni.wondr',       # App ID di Play Store
            lang='id',            # Bahasa Indonesia
            country='id',         # Region Indonesia
            sort=Sort.NEWEST,     # Ambil yang terbaru
            count=count
        )
        
        # Ekstrak hanya teks review-nya saja, bersihkan dari spasi berlebih
        review_texts = [r['content'].strip() for r in result if r['content']]
        logging.info(f"Berhasil mengambil {len(review_texts)} review.")
        
        return review_texts
    
    except Exception as e:
        logging.error(f"Gagal mengambil data dari Play Store: {e}")
        return []

if __name__ == "__main__":
    # Tes scraper
    data = get_wondr_reviews(5)
    for idx, text in enumerate(data):
        print(f"{idx+1}. {text}\n")