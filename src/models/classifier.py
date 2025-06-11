"""
Modelul de clasificare pentru companiile din industria de asigurări.
"""

import os
import joblib
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append('.')
from src.config.config import MODEL_NAME, BATCH_SIZE, THRESHOLD, MAX_LABELS, MODEL_CACHE_DIR


class InsuranceCompanyClassifier:
    """
    Clasificator pentru companiile din industria de asigurări.
    
    Utilizează modele de embeddings pentru a calcula similaritatea semantică 
    între descrierile companiilor și etichetele din taxonomie.
    """
    
    def __init__(self, 
                 model_name: str = MODEL_NAME,
                 threshold: float = THRESHOLD,
                 max_labels: int = MAX_LABELS,
                 cache_dir: str = MODEL_CACHE_DIR):
        """
        Inițializează clasificatorul.
        
        Args:
            model_name: Numele modelului SentenceTransformer
            threshold: Pragul de similaritate pentru clasificare
            max_labels: Numărul maxim de etichete per companie
            cache_dir: Directorul pentru cache-ul modelului
        """
        self.model_name = model_name
        self.threshold = threshold
        self.max_labels = max_labels
        
        # Crearea directorului cache dacă nu există
        os.makedirs(cache_dir, exist_ok=True)
        
        # Încărcarea modelului de embeddings
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        # Inițializare atribute
        self.taxonomy_labels = []
        self.taxonomy_embeddings = None
        self.similarities = None
        
        # Dicționar de cuvinte cheie pentru reguli de rezervă
        self.keyword_rules = {
            "general insurance": ["asigurări generale", "insurance", "asigurare", "asigurator", "insuring", "insurers", "coverage", "acoperire", "poliță", "policy", "policies", "insurance market", "piața asigurărilor"],
            "life insurance": ["asigurări de viață", "viață", "life insurance", "pensie", "retirement", "annuity", "anuitate", "life policy", "term insurance", "deces", "moștenire", "beneficiar", "beneficiary"],
            "health insurance": ["asigurări de sănătate", "sănătate", "health", "medical", "medicale", "sickness", "boală", "spital", "hospital", "ambulance", "ambulanță", "medical care", "îngrijire medicală"],
            "auto insurance": ["asigurări auto", "auto", "car", "mașină", "vehicul", "casco", "motor", "RCA", "traffic", "accident", "automobil", "automobile", "vehicle", "crash", "coliziune", "collision"],
            "property insurance": ["asigurări de proprietate", "proprietate", "property", "locuință", "home", "casă", "clădire", "building", "real estate", "imobil", "apartment", "apartament", "condominium", "damage", "daună"],
            "liability insurance": ["asigurări de răspundere", "răspundere", "liability", "responsabilitate", "responsibility", "third party", "terță parte", "damage claim", "compensație", "malpractice", "malpraxis"],
            "travel insurance": ["asigurări de călătorie", "călătorie", "travel", "turism", "tourism", "vacanță", "vacation", "holiday", "abroad", "străinătate", "trip", "journey", "voiaj"],
            "business insurance": ["asigurări pentru afaceri", "business", "company", "comercial", "corporate", "corporativ", "enterprise", "întreprindere", "business interruption", "întrerupere activitate", "commercial", "industrial"],
            "pet insurance": ["asigurări pentru animale", "animal", "pet", "animale de companie", "veterinary", "veterinar", "dog", "câine", "cat", "pisică"],
            "insurance broker": ["broker", "intermediar", "agent", "consultant", "consultanță", "intermediere", "brokerage", "brokeraj", "advisor", "agency", "agenție"],
            "claims management": ["managementul daunelor", "daune", "claim", "compensație", "compensation", "damage", "pagubă", "assessment", "evaluare", "settlement", "despăgubire"],
            "risk assessment": ["evaluare risc", "risc", "risk", "hazard", "pericol", "exposure", "expunere", "risk management", "managementul riscului", "prevention", "prevenție", "safety", "siguranță"],
            "reinsurance": ["reasigurare", "reinsurance", "cedare", "retrocession", "retrocesiune", "treaty", "tratat", "facultative", "facultativ"],
            "maritime insurance": ["asigurări maritime", "maritime", "naval", "ship", "navă", "vessel", "cargo", "marfă", "transport", "shipping", "marine"],
            "agricultural insurance": ["asigurări agricole", "agricol", "agricultural", "crops", "culturi", "livestock", "fermă", "farm", "farming", "harvest", "recoltă"],
            "cyber insurance": ["asigurări cyber", "cyber", "digital", "data", "date", "breach", "hack", "hacking", "IT", "technology", "tehnologie", "informatic"]
        }
    
    def load_taxonomy(self, taxonomy_path: str) -> List[str]:
        """
        Încarcă taxonomia din fișier.
        
        Args:
            taxonomy_path: Calea către fișierul cu taxonomia
            
        Returns:
            Lista de etichete din taxonomie
        """
        # Încărcare taxonomie
        taxonomy_df = pd.read_csv(taxonomy_path)
        self.taxonomy_labels = taxonomy_df['label'].tolist()
        
        return self.taxonomy_labels
    
    def compute_taxonomy_embeddings(self, cache_path: str = None) -> np.ndarray:
        """
        Calculează embeddings pentru etichetele din taxonomie.
        
        Args:
            cache_path: Calea pentru salvarea/încărcarea cache-ului (opțional)
            
        Returns:
            Matricea de embeddings pentru taxonomie
        """
        # Verificare dacă există cache
        if cache_path and os.path.exists(cache_path):
            self.taxonomy_embeddings = joblib.load(cache_path)
            return self.taxonomy_embeddings
        
        # Calcularea embeddings-urilor pentru taxonomie
        print(f"Calculare embeddings pentru {len(self.taxonomy_labels)} etichete din taxonomie...")
        self.taxonomy_embeddings = self.model.encode(
            self.taxonomy_labels, 
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Salvare cache dacă este specificat
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(self.taxonomy_embeddings, cache_path)
        
        return self.taxonomy_embeddings
    
    def compute_company_embeddings(self, 
                                  company_texts: List[str],
                                  cache_path: str = None) -> np.ndarray:
        """
        Calculează embeddings pentru textele companiilor.
        
        Args:
            company_texts: Lista de texte ale companiilor
            cache_path: Calea pentru salvarea/încărcarea cache-ului (opțional)
            
        Returns:
            Matricea de embeddings pentru companii
        """
        # Verificare dacă există cache
        if cache_path and os.path.exists(cache_path):
            return joblib.load(cache_path)
        
        # Calcularea embeddings-urilor pentru companii
        print(f"Calculare embeddings pentru {len(company_texts)} companii...")
        company_embeddings = self.model.encode(
            company_texts, 
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Salvare cache dacă este specificat
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(company_embeddings, cache_path)
        
        return company_embeddings
    
    def compute_similarities(self, 
                            company_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculează matricea de similarități între companii și taxonomie.
        
        Args:
            company_embeddings: Matricea de embeddings pentru companii
            
        Returns:
            Matricea de similarități
        """
        # Verificare dacă embeddings-urile taxonomiei sunt calculate
        if self.taxonomy_embeddings is None:
            raise ValueError("Embeddings-urile taxonomiei nu au fost calculate!")
        
        # Calcularea similarităților cosinus
        print("Calculare similarități între companii și taxonomie...")
        self.similarities = cosine_similarity(
            company_embeddings, 
            self.taxonomy_embeddings
        )
        
        return self.similarities
    
    def classify_companies(self, 
                          similarities: np.ndarray = None,
                          threshold: float = None,
                          max_labels: int = None) -> List[str]:
        """
        Clasifică companiile pe baza similarităților.
        
        Args:
            similarities: Matricea de similarități (opțional)
            threshold: Pragul de similaritate (opțional)
            max_labels: Numărul maxim de etichete (opțional)
            
        Returns:
            Lista de etichete pentru fiecare companie
        """
        # Utilizare valori implicite dacă parametrii nu sunt specificați
        if similarities is None:
            similarities = self.similarities
        if threshold is None:
            threshold = self.threshold
        if max_labels is None:
            max_labels = self.max_labels
        
        # Verificare dacă similaritățile sunt calculate
        if similarities is None:
            raise ValueError("Matricea de similarități nu a fost calculată!")
        
        # Inițializare listă pentru etichetele companiilor
        company_labels = []
        
        # Clasificare pentru fiecare companie
        for i in range(similarities.shape[0]):
            # Obținere scoruri de similaritate pentru compania curentă
            company_scores = similarities[i]
            
            # Filtrare etichete peste prag
            valid_indices = np.where(company_scores >= threshold)[0]
            
            # Sortare etichete după scor
            sorted_indices = valid_indices[np.argsort(company_scores[valid_indices])[::-1]]
            
            # Limitare la numărul maxim de etichete
            selected_indices = sorted_indices[:max_labels]
            
            # Creare șir de etichete
            if len(selected_indices) > 0:
                selected_labels = [self.taxonomy_labels[idx] for idx in selected_indices]
                company_labels.append(", ".join(selected_labels))
            else:
                company_labels.append("")
        
        return company_labels
    
    def apply_fallback_rules(self, 
                             company_data_list: List[Dict], 
                             company_labels: List[str]) -> List[str]:
        """
        Aplică reguli de rezervă pentru companiile fără etichete.
        
        Args:
            company_data_list: Lista cu datele originale ale companiilor
            company_labels: Lista de etichete atribuite prin similaritate
            
        Returns:
            Lista actualizată de etichete
        """
        updated_labels = company_labels.copy()
        
        for i, (data, labels) in enumerate(zip(company_data_list, company_labels)):
            # Dacă deja are etichete, continuă
            if labels:
                continue
                
            # Pregătește textul pentru căutare
            combined_text = ""
            
            # Adaugă descrierea dacă există
            if 'description' in data and isinstance(data['description'], str):
                combined_text += data['description'].lower() + " "
                
            # Adaugă business_tags dacă există
            if 'business_tags' in data:
                tags = data['business_tags']
                if isinstance(tags, list):
                    combined_text += " ".join(tags).lower() + " "
                elif isinstance(tags, str):
                    combined_text += tags.lower() + " "
                    
            # Adaugă alte câmpuri relevante
            for field in ['sector', 'category', 'niche', 'name', 'company_url', 'website']:
                if field in data and isinstance(data[field], str):
                    combined_text += data[field].lower() + " "
            
            # Îmbogățește textul cu sinonime și termeni relevanți
            enriched_text = combined_text + " insurance assurance risk policy coverage protection underwriting claim premium contract policyholder asigurări risc poliță acoperire protecție daună primă contract asigurat "
            
            # Aplică regulile bazate pe cuvinte cheie pe textul îmbogățit
            matched_labels = []
            
            for label, keywords in self.keyword_rules.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', enriched_text):
                        matched_labels.append(label)
                        break  # Odată ce am găsit o potrivire pentru această etichetă, nu mai căutăm
            
            # Analiză specifică sectorului
            if 'sector' in data and isinstance(data['sector'], str):
                sector = data['sector'].lower()
                # Mapare directă sector-etichetă
                sector_mapping = {
                    'insurance': 'general insurance',
                    'health': 'health insurance',
                    'healthcare': 'health insurance',
                    'financial': 'financial services, general insurance',
                    'fintech': 'financial services, general insurance',
                    'automotive': 'auto insurance',
                    'auto': 'auto insurance',
                    'car': 'auto insurance',
                    'travel': 'travel insurance',
                    'tourism': 'travel insurance',
                    'property': 'property insurance',
                    'real estate': 'property insurance',
                    'business': 'business insurance',
                    'enterprise': 'business insurance'
                }
                
                for key, value in sector_mapping.items():
                    if key in sector:
                        for label in value.split(', '):
                            if label not in matched_labels:
                                matched_labels.append(label)
            
            # Adaugă etichete găsite sau etichete generice ca ultimă soluție
            if matched_labels:
                updated_labels[i] = ", ".join(matched_labels[:self.max_labels])
            else:
                # Verificăm dacă e posibil o companie de asigurări
                insurance_keywords = ["insurance", "assurance", "asigurări", "asigurare", "insurer", "asigurator", 
                                      "broker", "underwriter", "subscriptor", "policy", "poliță", "protection", 
                                      "protecție", "financial", "financiar", "risk", "risc", "coverage", "acoperire",
                                      "premium", "primă", "claim", "daună", "compensation", "compensație"]
                
                for keyword in insurance_keywords:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text):
                        updated_labels[i] = "general insurance"
                        break
                
                # Determinare etichete pe baza numelui companiei
                if not updated_labels[i] and 'name' in data and isinstance(data['name'], str):
                    company_name = data['name'].lower()
                    if any(term in company_name for term in ["insur", "assur", "asigur", "cover", "protect", "broker", "risk", "risc"]):
                        updated_labels[i] = "general insurance"
                    elif any(term in company_name for term in ["life", "viata", "pension", "pensie"]):
                        updated_labels[i] = "life insurance"
                    elif any(term in company_name for term in ["health", "medical", "sanata", "sanat"]):
                        updated_labels[i] = "health insurance"
                    elif any(term in company_name for term in ["auto", "car", "vehicle", "vehicul", "motor"]):
                        updated_labels[i] = "auto insurance"
                    elif any(term in company_name for term in ["home", "house", "property", "casa", "locuinta", "proprietate"]):
                        updated_labels[i] = "property insurance"
                    elif any(term in company_name for term in ["travel", "trip", "journey", "calator"]):
                        updated_labels[i] = "travel insurance"
                
                # Ultimă soluție - toate companiile trebuie să aibă o etichetă
                if not updated_labels[i]:
                    # Alocăm o etichetă general insurance în acest caz
                    updated_labels[i] = "general insurance"
        
        return updated_labels
    
    def process_and_classify(self, 
                            company_texts: List[str],
                            company_data_list: List[Dict],
                            taxonomy_path: str,
                            company_embeddings_cache: str = None,
                            taxonomy_embeddings_cache: str = None) -> Tuple[List[str], np.ndarray]:
        """
        Procesează și clasifică companiile într-un singur flux de lucru.
        
        Args:
            company_texts: Lista de texte ale companiilor
            company_data_list: Lista cu datele originale ale companiilor
            taxonomy_path: Calea către fișierul cu taxonomia
            company_embeddings_cache: Calea pentru cache-ul embeddings-urilor companiilor
            taxonomy_embeddings_cache: Calea pentru cache-ul embeddings-urilor taxonomiei
            
        Returns:
            Tuple cu lista de etichete și matricea de similarități
        """
        # Încărcare taxonomie
        self.load_taxonomy(taxonomy_path)
        
        # Calculare embeddings pentru taxonomie
        self.compute_taxonomy_embeddings(taxonomy_embeddings_cache)
        
        # Calculare embeddings pentru companii
        company_embeddings = self.compute_company_embeddings(
            company_texts, 
            company_embeddings_cache
        )
        
        # Calculare similarități
        self.compute_similarities(company_embeddings)
        
        # Clasificare companii prin similaritate
        company_labels = self.classify_companies()
        
        # Aplică reguli de rezervă pentru companiile fără etichete
        company_labels = self.apply_fallback_rules(company_data_list, company_labels)
        
        return company_labels, self.similarities 