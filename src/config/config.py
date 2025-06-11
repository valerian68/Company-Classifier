"""
Fișier de configurare pentru clasificatorul de companii din industria de asigurări.
"""

# Căi de date
COMPANY_DATA_PATH = "data/ml_insurance_challenge.csv"
TAXONOMY_PATH = "data/insurance_taxonomy.csv"
OUTPUT_PATH = "output/classified_companies.csv"

# Configurare model
MODEL_NAME = "all-MiniLM-L6-v2"  # Model pentru embeddings
BATCH_SIZE = 32  # Dimensiune batch pentru procesare
THRESHOLD = 0.4  # Prag pentru clasificare - valoare mai mare pentru acuratețe maximă
MAX_LABELS = 5   # Număr maxim de etichete per companie

# Procesare text
TEXT_COLUMNS = ["description", "business_tags", "sector", "category", "niche"]
TEXT_WEIGHTS = {
    "description": 1.0,
    "business_tags": 0.8,
    "sector": 0.6,
    "category": 0.7,
    "niche": 0.7
}

# Modele și cache
MODEL_CACHE_DIR = "models/cache"
EMBEDDINGS_CACHE = "models/embeddings"

# Fișiere pentru cache
TAXONOMY_EMBEDDINGS_CACHE = f"{EMBEDDINGS_CACHE}/taxonomy_embeddings.pkl"
COMPANY_EMBEDDINGS_CACHE = f"{EMBEDDINGS_CACHE}/company_embeddings.pkl" 