import logging
import requests
import re
from bs4 import BeautifulSoup
from google_play_scraper import Sort, reviews as play_reviews

# Setup logger spesifik
logger = logging.getLogger(__name__)

def get_wondr_playstore_reviews(count=100) -> list[str]:
    """Mengambil review dari Google Play Store (Android)."""
    logger.info(f"[Play Store] Memulai pengambilan ulasan...")
    try:
        result, _ = play_reviews(
            'id.bni.wondr',
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=count
        )
        review_texts = [r['content'].strip() for r in result if r.get('content')]
        logger.info(f"[Play Store] ✅ Berhasil menarik {len(review_texts)} ulasan.")
        return review_texts
    except Exception as e:
        logger.error(f"[Play Store] ❌ Gagal menarik data: {e}")
        return []

def get_wondr_appstore_reviews(count=100) -> list[str]:
    """
    Mengambil review App Store menggunakan teknik DOM Parsing (BeautifulSoup).
    Mengekstrak langsung ulasan dari HTML statis yang dikirim Apple.
    """
    logger.info(f"[App Store] Membuka web Apple dan membedah HTML...")
    review_texts = []
    app_id = '6499518320'
    country = 'id'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    try:
        # 1. Unduh halaman web Apple secara langsung
        web_url = f"https://apps.apple.com/{country}/app/wondr-by-bni/id{app_id}"
        response = requests.get(web_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"[App Store] Gagal mengakses web. Status: {response.status_code}")
            return []
            
        # 2. Bedah HTML menggunakan BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. Cari container ulasan berdasarkan aria-labelledby
        review_divs = soup.find_all('div', attrs={'aria-labelledby': re.compile(r'^review-')})
        
        if not review_divs:
            logger.error("[App Store] ❌ Tidak ada elemen ulasan yang ditemukan di HTML.")
            return []
            
        for div in review_divs:
            # --- PERBAIKAN PARSING DI SINI ---
            # Cari elemen Judul (<h3>)
            title_tag = div.find('h3', class_=re.compile(r'title'))
            # Cari elemen Isi Komentar (<p data-testid="truncate-text">)
            content_tag = div.find('p', attrs={'data-testid': 'truncate-text'})
            
            title = title_tag.get_text(strip=True) if title_tag else ""
            content = content_tag.get_text(strip=True) if content_tag else ""
            
            # Format: "Judul - Isi Review"
            full_text = f"{title} - {content}" if title else content
            
            if full_text:
                review_texts.append(full_text)
                
            if len(review_texts) >= count:
                break
                
        logger.info(f"[App Store] ✅ Berhasil membedah dan menarik {len(review_texts)} ulasan dari HTML.")
        return review_texts
        
    except Exception as e:
        logger.error(f"[App Store] ❌ Terjadi error: {e}")
        return review_texts

# --- Blok Testing Lokal ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n--- Uji Coba Scraper Play Store ---")
    play_data = get_wondr_playstore_reviews(3)
    for i, text in enumerate(play_data, 1): print(f"{i}. {text}")
        
    print("\n--- Uji Coba Scraper App Store ---")
    app_data = get_wondr_appstore_reviews(3)
    for i, text in enumerate(app_data, 1): print(f"{i}. {text}")
    print("\n")