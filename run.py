
"""Script principal pentru rularea clasificatorului de companii din industria de asigurări."""

import os
import argparse
import time
import shutil
from datetime import datetime
from colorama import init, Fore, Style
from tqdm.auto import tqdm

from src.config.config import (
    COMPANY_DATA_PATH, 
    TAXONOMY_PATH, 
    OUTPUT_PATH, 
    MODEL_NAME, 
    THRESHOLD, 
    MAX_LABELS,
    BATCH_SIZE
)
from src.models.classifier import InsuranceCompanyClassifier
from src.models.batch_processor import BatchProcessor


# Inițializare colorama pentru colorarea textului în terminal
init(autoreset=True)


def print_header(text):
    """Afișează un header formatat frumos în terminal."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(80)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 80}{Style.RESET_ALL}\n")


def print_subheader(text):
    """Afișează un subheader formatat frumos în terminal."""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{text}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{'-' * len(text)}{Style.RESET_ALL}\n")


def print_info(text):
    """Afișează informații formatate frumos în terminal."""
    print(f"{Fore.WHITE}{text}")


def print_success(text):
    """Afișează mesaje de succes formatate frumos în terminal."""
    print(f"{Fore.GREEN}{text}")


def print_warning(text):
    """Afișează avertismente formatate frumos în terminal."""
    print(f"{Fore.YELLOW}{text}")


def print_error(text):
    """Afișează erori formatate frumos în terminal."""
    print(f"{Fore.RED}{text}")


def parse_arguments():
    """Parsare argumente linie de comandă."""
    parser = argparse.ArgumentParser(
        description='Clasificator de companii pentru industria de asigurări',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--clean', 
        action='store_true',
        help='Șterge rezultatele și cache-ul anterior, fără a rula clasificarea'
    )
    
    return parser.parse_args()


def clean_previous_results(verbose=True):
    """Șterge rezultatele și cache-ul anterior."""
    paths_to_clean = [
        'output',
        'models/cache'
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            print_info(f"Ștergere {path}...")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print_success(f"Director {path} șters cu succes.")
            except Exception as e:
                print_error(f"Eroare la ștergerea {path}: {str(e)}")
    
    # Recreare directoare necesare
    for path in paths_to_clean:
        os.makedirs(path, exist_ok=True)
    
    print_success("Curățare completă. Directoarele au fost recreate.")


def run_classification():
    """Rulează clasificatorul complet."""
    # Creare directoare necesare
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs('models/cache', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    
    # Creare clasificator
    print_info(f"Inițializare clasificator cu modelul {MODEL_NAME}...")
    classifier = InsuranceCompanyClassifier(
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        max_labels=MAX_LABELS,
        cache_dir='models/cache'
    )
    
    # Creare procesor de batch-uri
    processor = BatchProcessor(
        classifier=classifier,
        batch_size=BATCH_SIZE
    )
    
    # Definire căi pentru cache
    cache_dir = 'models'
    
    # Procesare date
    print_subheader("Procesare date")
    start_time = time.time()
    
    # Procesare fișier
    from src.config.config import TEXT_COLUMNS, TEXT_WEIGHTS
    result_df = processor.process_file(
        input_path=COMPANY_DATA_PATH,
        taxonomy_path=TAXONOMY_PATH,
        output_path=OUTPUT_PATH,
        text_columns=TEXT_COLUMNS,
        text_weights=TEXT_WEIGHTS,
        output_column='insurance_label',
        cache_dir=cache_dir
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Afișare informații despre procesare
    print_success(f"\nProcesare completă în {processing_time:.2f} secunde.")
    print_success(f"Rezultate salvate în: {OUTPUT_PATH}")
    
    # Generare grafice și metrici
    print_subheader("Generare grafice și metrici de evaluare")
    
    # Importăm aici pentru a evita încărcarea inutilă dacă nu sunt necesare
    from src.utils.evaluation import plot_similarity_distribution, plot_top_labels, print_evaluation_metrics
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    
    # Creare director pentru grafice
    plots_dir = os.path.join(os.path.dirname(OUTPUT_PATH), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generare timestamp pentru nume fișiere
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generare grafice de bază
    similarity_plot_path = os.path.join(plots_dir, f'similarity_distribution_{timestamp}.png')
    labels_plot_path = os.path.join(plots_dir, f'top_labels_{timestamp}.png')
    
    # Presupunem că avem acces la similarități prin clasificator
    if classifier.similarities is not None:
        # Grafic de distribuție a similarităților
        plot_similarity_distribution(
            classifier.similarities, 
            THRESHOLD, 
            similarity_plot_path
        )
        print_info(f"Grafic distribuție similarități salvat în: {similarity_plot_path}")
        
        # Matrice de căldură pentru top similarități
        print_info("Generare matrice de căldură pentru top similarități...")
        plt.figure(figsize=(12, 10))
        
        # Extrage top 20 etichete după frecvență
        label_counts = {}
        for label_str in result_df['insurance_label']:
            if isinstance(label_str, str) and label_str.strip():
                for label in label_str.split(', '):
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        top_label_names = [label for label, _ in top_labels]
        
        # Extrage top 15 companii care au primit etichete
        companies_with_labels = result_df[result_df['insurance_label'].notna() & 
                                          (result_df['insurance_label'] != '')].iloc[:15]
        
        # Construiește o matrice de similaritate pentru vizualizare
        if len(companies_with_labels) > 0 and len(top_label_names) > 0:
            # Găsește indicii etichetelor în taxonomy_labels
            label_indices = [classifier.taxonomy_labels.index(label) 
                            for label in top_label_names 
                            if label in classifier.taxonomy_labels]
            
            # Construiește submatricea de similaritate pentru top companii și top etichete
            if len(label_indices) > 0:
                company_indices = companies_with_labels.index.tolist()
                company_indices = [i for i in company_indices if i < classifier.similarities.shape[0]]
                
                if len(company_indices) > 0:
                    sub_matrix = classifier.similarities[company_indices][:, label_indices]
                    
                    # Creează heatmap
                    ax = sns.heatmap(sub_matrix, cmap="YlGnBu", annot=True, fmt=".2f",
                                    xticklabels=[classifier.taxonomy_labels[i] for i in label_indices],
                                    yticklabels=[f"Comp {i+1}" for i in range(len(company_indices))])
                    
                    plt.title("Matrice de similaritate pentru top companii și etichete")
                    plt.tight_layout()
                    
                    heatmap_path = os.path.join(plots_dir, f'similarity_heatmap_{timestamp}.png')
                    plt.savefig(heatmap_path)
                    plt.close()
                    print_info(f"Matrice de căldură salvată în: {heatmap_path}")
    
    # Grafic cu top etichete
    plot_top_labels(
        result_df, 
        top_n=20, 
        label_column='insurance_label',
        save_path=labels_plot_path
    )
    print_info(f"Grafic top etichete salvat în: {labels_plot_path}")
    
    # Grafic de performanță pentru etichete
    print_info("Generare grafic de performanță pentru etichete...")
    
    # Extrage date pentru grafic
    label_counts = {}
    total_companies = len(result_df)
    labeled_companies = sum(1 for x in result_df['insurance_label'] 
                            if isinstance(x, str) and x.strip())
    
    # Calculează procentul de companii etichetate
    labeled_percentage = labeled_companies / total_companies * 100
    
    # Creează grafic
    plt.figure(figsize=(10, 6))
    plt.bar(['Etichetate', 'Neetichetate'], 
            [labeled_percentage, 100 - labeled_percentage],
            color=['green', 'red'])
    plt.title('Procentaj de companii etichetate')
    plt.ylabel('Procent (%)')
    plt.ylim(0, 100)
    
    # Adaugă etichete cu numărul exact de companii
    for i, v in enumerate([labeled_companies, total_companies - labeled_companies]):
        plt.text(i, 5, f"{v} companii", ha='center')
    
    performance_path = os.path.join(plots_dir, f'performance_chart_{timestamp}.png')
    plt.savefig(performance_path)
    plt.close()
    print_info(f"Grafic de performanță salvat în: {performance_path}")
    
    # Afișare metrici de evaluare
    if classifier.similarities is not None:
        metrics = print_evaluation_metrics(
            result_df, 
            classifier.similarities, 
            THRESHOLD,
            'insurance_label'
        )
    
    return result_df


def main():
    """Funcția principală pentru clasificare."""
    # Parsare argumente
    args = parse_arguments()
    
    print_header("CLASIFICATOR DE COMPANII PENTRU INDUSTRIA DE ASIGURĂRI")
    
    # Dacă --clean este specificat, doar șterge rezultatele anterioare și oprește-te
    if args.clean:
        print_subheader("Curățare rezultate anterioare")
        clean_previous_results()
        print_success("Curățare completă. Folosiți 'python run.py' pentru a rula clasificarea.")
        return
    
    # Afișare configurație
    print_subheader("Configurație")
    print_info(f"Fișier date companii: {COMPANY_DATA_PATH}")
    print_info(f"Fișier taxonomie: {TAXONOMY_PATH}")
    print_info(f"Fișier output: {OUTPUT_PATH}")
    print_info(f"Model: {MODEL_NAME}")
    print_info(f"Prag similaritate: {THRESHOLD}")
    print_info(f"Număr maxim etichete: {MAX_LABELS}")
    print_info(f"Dimensiune batch: {BATCH_SIZE}")
    
    # Rulare clasificare
    run_classification()
    
    print_header("PROCESARE FINALIZATĂ")


if __name__ == "__main__":
    main() 