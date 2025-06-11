"""
Funcții pentru evaluarea performanței clasificatorului.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def compute_similarity_distribution(similarities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculează distribuția valorilor de similaritate.
    
    Args:
        similarities: Matricea de similarități
        
    Returns:
        Tuple cu valorile binilor histogramei și frecvențele
    """
    # Aplatizează matricea de similarități
    flat_similarities = similarities.flatten()
    
    # Calculul histogramei
    hist, bin_edges = np.histogram(flat_similarities, bins=20, range=(0, 1))
    
    return bin_edges[:-1], hist


def plot_similarity_distribution(similarities: np.ndarray, 
                                 threshold: float = 0.5, 
                                 save_path: str = None) -> None:
    """
    Generează un grafic cu distribuția similarităților.
    
    Args:
        similarities: Matricea de similarități
        threshold: Pragul de clasificare
        save_path: Calea pentru salvarea graficului (opțional)
    """
    bin_edges, hist = compute_similarity_distribution(similarities)
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges, hist, width=0.05, alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Prag: {threshold}')
    
    plt.xlabel('Scor Similaritate')
    plt.ylabel('Frecvență')
    plt.title('Distribuția Scorurilor de Similaritate')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def compute_label_distribution(df: pd.DataFrame, 
                               label_column: str = 'insurance_label') -> Tuple[List[str], List[int]]:
    """
    Calculează distribuția etichetelor atribuite.
    
    Args:
        df: DataFrame cu datele companiilor
        label_column: Numele coloanei cu etichete
        
    Returns:
        Tuple cu etichetele și numărul de apariții
    """
    # Extragere și numărare etichete
    all_labels = []
    for labels_str in df[label_column]:
        if isinstance(labels_str, str):
            labels = labels_str.split(', ')
            all_labels.extend(labels)
    
    # Numărare apariții
    counter = Counter(all_labels)
    labels, counts = zip(*counter.most_common())
    
    return labels, counts


def plot_top_labels(df: pd.DataFrame, 
                    top_n: int = 20, 
                    label_column: str = 'insurance_label',
                    save_path: str = None) -> None:
    """
    Generează un grafic cu top N etichete atribuite.
    
    Args:
        df: DataFrame cu datele companiilor
        top_n: Numărul de etichete afișate
        label_column: Numele coloanei cu etichete
        save_path: Calea pentru salvarea graficului (opțional)
    """
    labels, counts = compute_label_distribution(df, label_column)
    
    # Limitare la top N
    if len(labels) > top_n:
        labels = labels[:top_n]
        counts = counts[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(labels)), counts, align='center')
    plt.yticks(range(len(labels)), labels)
    
    plt.xlabel('Număr de Companii')
    plt.ylabel('Etichete')
    plt.title(f'Top {len(labels)} Etichete Atribuite')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def compute_average_labels_per_company(df: pd.DataFrame, 
                                       label_column: str = 'insurance_label') -> float:
    """
    Calculează numărul mediu de etichete per companie.
    
    Args:
        df: DataFrame cu datele companiilor
        label_column: Numele coloanei cu etichete
        
    Returns:
        Numărul mediu de etichete per companie
    """
    label_counts = []
    
    for labels_str in df[label_column]:
        if isinstance(labels_str, str) and labels_str.strip():
            labels = labels_str.split(', ')
            label_counts.append(len(labels))
        else:
            label_counts.append(0)
    
    return np.mean(label_counts)


def print_evaluation_metrics(df: pd.DataFrame, 
                             similarities: np.ndarray,
                             threshold: float,
                             label_column: str = 'insurance_label') -> Dict[str, Any]:
    """
    Afișează și returnează metricile de evaluare pentru model.
    
    Args:
        df: DataFrame cu datele companiilor
        similarities: Matricea de similarități
        threshold: Pragul de clasificare
        label_column: Numele coloanei cu etichete
        
    Returns:
        Dicționar cu metricile calculate
    """
    # Calcularea metricilor
    avg_labels = compute_average_labels_per_company(df, label_column)
    no_label_count = sum(1 for x in df[label_column] if not isinstance(x, str) or not x.strip())
    classification_rate = 1 - (no_label_count / len(df))
    
    # Distribuția similarităților
    above_threshold = (similarities >= threshold).sum()
    total_comparisons = similarities.size
    above_threshold_pct = above_threshold / total_comparisons * 100
    
    # Crearea dicționarului de metrici
    metrics = {
        'avg_labels_per_company': avg_labels,
        'classification_rate': classification_rate,
        'above_threshold_pct': above_threshold_pct,
        'company_count': len(df),
        'no_label_count': no_label_count
    }
    
    # Afișare metrici
    print("\n===== Metrici de Evaluare =====")
    print(f"Număr total de companii: {len(df)}")
    print(f"Rata de clasificare: {classification_rate:.2%}")
    print(f"Etichete medii per companie: {avg_labels:.2f}")
    print(f"Companii fără etichetă: {no_label_count} ({(no_label_count / len(df)):.2%})")
    print(f"Procent similarități peste prag ({threshold}): {above_threshold_pct:.2f}%")
    print("================================\n")
    
    return metrics 