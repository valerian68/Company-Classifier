"""
Procesare în batch-uri pentru clasificatorul de companii.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

import sys
sys.path.append('.')
from src.config.config import BATCH_SIZE, TEXT_COLUMNS, TEXT_WEIGHTS
from src.utils.text_processing import prepare_company_data
from src.models.classifier import InsuranceCompanyClassifier


class BatchProcessor:
    """
    Procesează datele în batch-uri pentru a optimiza memoria și timpul de execuție.
    """
    
    def __init__(self, 
                 classifier: InsuranceCompanyClassifier,
                 batch_size: int = BATCH_SIZE):
        """
        Inițializează procesorul de batch-uri.
        
        Args:
            classifier: Clasificatorul de companii
            batch_size: Dimensiunea batch-ului
        """
        self.classifier = classifier
        self.batch_size = batch_size
    
    def process_dataframe(self, 
                         df: pd.DataFrame,
                         taxonomy_path: str,
                         text_columns: List[str] = TEXT_COLUMNS,
                         text_weights: Dict[str, float] = TEXT_WEIGHTS,
                         output_column: str = 'insurance_label',
                         cache_dir: str = 'models') -> pd.DataFrame:
        """
        Procesează un DataFrame în batch-uri.
        
        Args:
            df: DataFrame cu datele companiilor
            taxonomy_path: Calea către fișierul cu taxonomia
            text_columns: Coloanele de text pentru procesare
            text_weights: Ponderile pentru fiecare câmp de text
            output_column: Numele coloanei pentru rezultate
            cache_dir: Directorul pentru cache
            
        Returns:
            DataFrame cu rezultatele clasificării
        """
        # Încărcare taxonomie
        self.classifier.load_taxonomy(taxonomy_path)
        
        # Calculare embeddings pentru taxonomie
        taxonomy_cache = os.path.join(cache_dir, 'taxonomy_embeddings.pkl')
        self.classifier.compute_taxonomy_embeddings(taxonomy_cache)
        
        # Pregătire date
        print("Pregătire date pentru procesare...")
        combined_texts, company_data_list = prepare_company_data(df, text_columns, text_weights)
        
        # Calculare număr de batch-uri
        n_samples = len(combined_texts)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Inițializare listă pentru rezultate
        all_labels = []
        all_batch_data = []  # Stocăm datele originale pentru fiecare batch
        
        # Procesare batch-uri
        print(f"Procesare {n_samples} companii în {n_batches} batch-uri...")
        start_time = time.time()
        
        for i in tqdm(range(n_batches), desc="Batches"):
            # Extragere date pentru batch-ul curent
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch_texts = combined_texts[start_idx:end_idx]
            batch_data = company_data_list[start_idx:end_idx]
            
            # Calculare embeddings pentru batch
            batch_cache = None  # Nu folosim cache pentru batch-uri
            batch_embeddings = self.classifier.compute_company_embeddings(batch_texts, batch_cache)
            
            # Calculare similarități pentru batch
            batch_similarities = self.classifier.compute_similarities(batch_embeddings)
            
            # Clasificare batch prin similaritate
            batch_labels = self.classifier.classify_companies(batch_similarities)
            
            # Aplicare reguli de rezervă pentru companiile fără etichete
            batch_labels = self.classifier.apply_fallback_rules(batch_data, batch_labels)
            
            # Adăugare rezultate la lista completă
            all_labels.extend(batch_labels)
            all_batch_data.extend(batch_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Procesare completă în {processing_time:.2f} secunde "
              f"({n_samples / processing_time:.2f} companii/secundă)")
        
        # Adăugare rezultate la DataFrame
        result_df = df.copy()
        result_df[output_column] = all_labels
        
        # Afișare statistici
        unlabeled_count = sum(1 for label in all_labels if not label)
        print(f"Statistici clasificare:")
        print(f"  - Total companii: {n_samples}")
        print(f"  - Companii etichetate: {n_samples - unlabeled_count} ({(n_samples - unlabeled_count) / n_samples * 100:.2f}%)")
        print(f"  - Companii fără etichete: {unlabeled_count} ({unlabeled_count / n_samples * 100:.2f}%)")
        
        return result_df
    
    def process_file(self, 
                    input_path: str,
                    taxonomy_path: str,
                    output_path: str,
                    text_columns: List[str] = TEXT_COLUMNS,
                    text_weights: Dict[str, float] = TEXT_WEIGHTS,
                    output_column: str = 'insurance_label',
                    cache_dir: str = 'models') -> pd.DataFrame:
        """
        Procesează un fișier CSV în batch-uri.
        
        Args:
            input_path: Calea către fișierul de intrare
            taxonomy_path: Calea către fișierul cu taxonomia
            output_path: Calea pentru fișierul de ieșire
            text_columns: Coloanele de text pentru procesare
            text_weights: Ponderile pentru fiecare câmp de text
            output_column: Numele coloanei pentru rezultate
            cache_dir: Directorul pentru cache
            
        Returns:
            DataFrame cu rezultatele clasificării
        """
        # Încărcare date
        print(f"Încărcare date din {input_path}...")
        df = pd.read_csv(input_path)
        
        # Procesare date
        result_df = self.process_dataframe(
            df, 
            taxonomy_path, 
            text_columns, 
            text_weights, 
            output_column, 
            cache_dir
        )
        
        # Creare director pentru output dacă nu există
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salvare rezultate
        print(f"Salvare rezultate în {output_path}...")
        result_df.to_csv(output_path, index=False)
        
        return result_df 