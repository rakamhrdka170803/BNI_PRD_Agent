# BNI PRD Agent

AI Product Manager Agent untuk membantu menyusun Product Requirements Document (PRD) berbasis Groq LLM, dengan konteks tambahan dari user feedback wondr by BNI (Play Store) yang dianalisis sentimen (IndoBERT) dan disimpan sebagai vektor di PostgreSQL/pgvector.

## Arsitektur

- **App** - Gradio UI (`app.py`) di port `7860`
- **LLM** - Groq (default `qwen/qwen3-32b`) via `langchain-groq`
- **Embeddings** - Ollama (default `qwen3-embedding:4b`, dimensi 2560) di port `11434`
- **Vector DB** - PostgreSQL 16 + pgvector di port `5432`
- **Sentiment** - IndoBERT (`w11wo/indonesian-roberta-base-sentiment-classifier`) via `transformers`
- **Observability** - MLflow tracing & prompt registry via `bni-llmops`

## Prasyarat

- Docker + Docker Compose, atau Python 3.11 + PostgreSQL 16 (pgvector) + Ollama lokal
- Akun & API key Groq (https://console.groq.com)
- Akses ke repo `gendonholaholo/bni-mlops-mlflow` (terinstall otomatis dari `requirements.txt`)

## Setup

### 1. Konfigurasi environment

```bash
cp .env.example .env
```

Edit `.env` dan isi `GROQ_API_KEY`. Variabel penting:

| Variabel | Default | Keterangan |
|---|---|---|
| `GROQ_API_KEY` | - | Wajib diisi |
| `GROQ_MODEL` | `qwen/qwen3-32b` | Nama model di Groq |
| `DB_HOST` / `DB_PORT` | `localhost` / `5432` | PostgreSQL |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:4b` | Model embedding (dim 2560) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow server |

## Cara menjalankan

### Opsi A - Docker Compose (rekomendasi)

```bash
docker compose up -d --build
```

Service yang naik: `prd_postgres`, `prd_ollama`, `prd_agent_app`. Setelah container hidup, jalankan langkah inisialisasi sekali (lihat bagian "Inisialisasi data & prompt" di bawah).

UI tersedia di http://localhost:7860.

### Opsi B - Jalan lokal (tanpa Docker)

1. Siapkan PostgreSQL 16 dengan ekstensi pgvector dan database `prd_db`.
2. Install & jalankan Ollama, lalu pull model embedding:

   ```bash
   ollama pull qwen3-embedding:4b
   ```

3. Buat virtualenv dan install dependency:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Jalankan langkah inisialisasi (di bawah).
5. Jalankan app:

   ```bash
   python app.py
   ```

   Buka http://localhost:7860.

## Inisialisasi data & prompt (sekali saja)

Jalankan dari host (atau `docker compose exec app ...`):

1. **Register prompt PRD ke MLflow registry** (alias `production`):

   ```bash
   python register_prompt.py
   ```

2. **Seed database** - scrape 200 review terbaru wondr by BNI, jalankan IndoBERT untuk sentimen, embed dengan Ollama, simpan ke pgvector:

   ```bash
   python init_database_schema.py
   ```

   Catatan: pertama kali jalan akan mengunduh model IndoBERT (~400MB). Skrip ini melakukan `DROP TABLE IF EXISTS user_feedback` lalu membuat ulang skema, jadi semua data lama akan hilang.

3. (Opsional) Export isi DB ke CSV untuk inspeksi:

   ```bash
   python cek_data.py
   ```

## Alur penggunaan UI

1. Isi 6 field di kolom kiri (Objective, Target Users, Pain Points, Features, Acceptance Criteria, Out of Scope).
2. Klik **Generate PRD** - agen menarik feedback relevan dari pgvector (RAG dengan prioritas Negatif/Netral lalu Positif, threshold similarity 0.6) dan menghasilkan draft Markdown.
3. Iterasi via **Submit Revisi** (history percakapan dipelihara di `gr.State`).
4. Klik **Approve & Generate PDF** untuk mengekspor `Approved_PRD.pdf`.

## Struktur kode

| File | Peran |
|---|---|
| `app.py` | UI Gradio + wiring event |
| `agent_logic.py` | `PRDAgent` - generate & revise via Groq |
| `retriever.py` | `FeedbackRetriever` - RAG SQL ke pgvector |
| `scraper.py` | Ambil review Play Store via `google-play-scraper` |
| `init_database_schema.py` | Setup skema + seed data (sentimen + embedding) |
| `register_prompt.py` | Daftarkan template prompt ke MLflow |
| `cek_data.py` | Export tabel `user_feedback` ke CSV |
| `docker-compose.yaml` | Definisi 3 service (db, ollama, app) |
| `DockerFile` | Image Python 3.11-slim untuk app |

## Troubleshooting

- **`Gagal terkoneksi ke database`** - pastikan PostgreSQL hidup dan `DB_HOST` benar. Jika pakai Docker Compose dari dalam container app, gunakan `DB_HOST=db`.
- **Embedding error / dimensi mismatch** - kolom `embedding` dibuat dengan `VECTOR(2560)` mengikuti `qwen3-embedding:4b`. Jika ganti model embedding, update `VECTOR_DIMENSIONS` di `.env` dan `init_database_schema.py`, lalu jalankan ulang seeding.
- **Ollama 404 model** - jalankan `ollama pull qwen3-embedding:4b` (atau model yang dikonfigurasi di `OLLAMA_EMBED_MODEL`).
- **MLflow tracing tidak muncul** - cek `MLFLOW_TRACKING_URI` dan pastikan server MLflow jalan; untuk mematikan tracing set `LLMOPS_DISABLE_TRACING=true`.
