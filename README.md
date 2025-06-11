## Funcționalități

- Clasificare semantica bazata pe modele de embedding (Sentence-Transformers)
- Procesare optimizata in batch-uri pentru seturi mari de date (<2 minute)
- Sistem de cache pentru accelerare rulari ulterioare
- Vizualizari avansate pentru analiza rezultatelor
- Interfata CLI cu afisare color pentru monitorizare

## Structura Proiectului

```
.
├── data/                     # Date de intrare
├── models/                   # Cache pentru embeddings
├── output/                   # Rezultate si vizualizari
└── src/                      # Cod sursa
    ├── config/               # Parametri configurabili
    ├── models/               # Modele de clasificare
    └── utils/                # Utilitare pentru procesare si evaluare
```

## Instalare

```bash
# 1. Clonare repository
git clone <repository-url>
cd clasificator-companii
# 2. Creare mediu virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Instalare dependente
pip install -r requirements.txt
```

## Utilizare

Rulare clasificare cu parametrii impliciti:

```bash
python run.py
```

Curatare rezultate anterioare:

```bash
python run.py --clean
```

## Procesul de Clasificare

1. **Preprocesare**: Combinarea si ponderarea diferitelor campuri text ale companiei
2. **Vectorizare**: Transformarea textelor in embeddings vectoriale folosind Sentence-Transformers
3. **Similaritate**: Calculul similaritatii cosinus intre companii si etichetele taxonomice
4. **Clasificare**: Atribuirea etichetelor cu cel mai mare scor de similaritate (prag: 0.1)
5. **Evaluare**: Generarea metricilor si vizualizarilor pentru analiza rezultatelor

## Rezultate

Sistemul genereaza:

- Fisier CSV cu companiile clasificate si etichetele atribuite
- Vizualizari (distributia similaritatilor, matrice de caldura, frecventa etichetelor)
- Metrici de evaluare (rata de clasificare, numar mediu de etichete/companie)

Optimizat pentru:

- Eficienta: procesare in batch-uri pentru economisire de memorie
- Viteza: cache pentru embeddings si paralelizare
- Acuratete: ponderare inteligenta a campurilor text

## Autor

Ursu Valerian
