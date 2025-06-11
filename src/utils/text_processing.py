"""
Funcții pentru procesarea textului și pregătirea datelor pentru clasificare.
"""

import re
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple


def clean_text(text: str) -> str:
    """
    Curăță textul de caractere speciale și formatări.
    
    Args:
        text: Textul care trebuie curățat
        
    Returns:
        Textul curățat
    """
    if not isinstance(text, str):
        return ""
    
    # Înlocuire caractere non-alfanumerice cu spații
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Înlocuire spații multiple cu un singur spațiu
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminare spații de la început și sfârșit
    return text.strip().lower()


def parse_business_tags(tags: str) -> List[str]:
    """
    Parsează șirul de business_tags în lista de tag-uri.
    
    Args:
        tags: Șirul de tag-uri în format '[tag1, tag2, ...]'
        
    Returns:
        Lista de tag-uri curățate
    """
    if not isinstance(tags, str):
        return []
    
    try:
        # Încercare de parsare ca listă Python
        tag_list = ast.literal_eval(tags)
        if isinstance(tag_list, list):
            return [clean_text(tag) for tag in tag_list]
    except (ValueError, SyntaxError):
        # Dacă parsarea eșuează, încercăm să splitam după virgulă
        return [clean_text(tag) for tag in tags.split(',')]
    
    return []


def combine_company_text(company_data: Dict[str, Union[str, List[str]]], 
                         weights: Dict[str, float]) -> str:
    """
    Combină textele disponibile pentru o companie într-un text unic ponderat.
    Îmbogățește textul cu termeni relevanți din domeniul asigurărilor.
    
    Args:
        company_data: Dicționar cu datele companiei
        weights: Ponderile pentru fiecare câmp de text
        
    Returns:
        Text combinat pentru clasificare
    """
    # Dicționar de termeni și sinonime din domeniul asigurărilor
    insurance_terms = {
        "asigurare": "asigurare insurance polița policy acoperire coverage protecție protection",
        "risc": "risc risk hazard pericol danger",
        "dauna": "dauna claim paguba damage pierdere loss compensație compensation",
        "accident": "accident eveniment event incident nefericit unfortunate",
        "proprietate": "proprietate property bun asset posesiune possession",
        "sănătate": "sănătate health medical medical îngrijire care",
        "viață": "viață life protecție protection economii savings",
        "auto": "auto car vehicul vehicle automobil automobile mașină",
        "locuință": "locuință home casă house apartament apartment imobil building",
        "răspundere": "răspundere liability responsabilitate responsibility",
        "pensie": "pensie pension retragere retirement bătrânețe old-age",
        "investiție": "investiție investment plasament placement fond fund",
        "broker": "broker intermediar intermediary agent agent consultant consultant",
        "primă": "primă premium cost cost plată payment",
        "indemnizație": "indemnizație indemnity compensație compensation beneficiu benefit",
        "acoperire": "acoperire coverage protecție protection garantare guarantee",
        "reasigurare": "reasigurare reinsurance transfer transfer distribuție distribution",
        "evaluare": "evaluare assessment estimare estimation analiză analysis",
        "subscriere": "subscriere underwriting acceptare acceptance"
    }
    
    combined_text = []
    
    # Adăugare descriere îmbogățită
    if 'description' in company_data and company_data['description']:
        desc = company_data['description']
        # Îmbogățim textul cu termeni din domeniul asigurărilor
        enriched_desc = desc
        for term, synonyms in insurance_terms.items():
            if term in desc.lower():
                enriched_desc = f"{enriched_desc} {synonyms}"
        
        combined_text.append(enriched_desc * int(weights.get('description', 1) * 10))
    
    # Adăugare business tags îmbogățite
    if 'business_tags' in company_data and company_data['business_tags']:
        tags = company_data['business_tags']
        if isinstance(tags, list):
            tags_text = ' '.join(tags)
        else:
            tags_text = tags
            
        # Îmbogățim tag-urile
        enriched_tags = tags_text
        for term, synonyms in insurance_terms.items():
            if term in tags_text.lower():
                enriched_tags = f"{enriched_tags} {synonyms}"
                
        combined_text.append(enriched_tags * int(weights.get('business_tags', 0.8) * 10))
    
    # Adăugare sector, categorie și nișă îmbogățite
    for field in ['sector', 'category', 'niche']:
        if field in company_data and company_data[field]:
            field_text = company_data[field]
            
            # Îmbogățim câmpul cu termeni din domeniul asigurărilor
            enriched_field = field_text
            for term, synonyms in insurance_terms.items():
                if term in field_text.lower():
                    enriched_field = f"{enriched_field} {synonyms}"
                    
            combined_text.append(enriched_field * int(weights.get(field, 0.5) * 10))
    
    # Adăugăm termeni generali din domeniul asigurărilor pentru a crește șansele de potrivire
    combined_text.append("asigurare insurance protecție risc acoperire daună claim")
    
    return ' '.join(combined_text)


def prepare_company_data(df: pd.DataFrame, text_columns: List[str], 
                         weights: Dict[str, float]) -> Tuple[List[str], List[Dict]]:
    """
    Pregătește datele companiilor pentru clasificare.
    
    Args:
        df: DataFrame cu datele companiilor
        text_columns: Coloanele de text care trebuie procesate
        weights: Ponderile pentru fiecare câmp de text
        
    Returns:
        Tuple cu lista de texte combinate și lista de date originale
    """
    combined_texts = []
    company_data_list = []
    
    for _, row in df.iterrows():
        company_data = {}
        
        # Extragere și curățare date
        for col in text_columns:
            if col in df.columns:
                if col == 'business_tags':
                    company_data[col] = parse_business_tags(row[col])
                else:
                    company_data[col] = clean_text(str(row[col]))
            else:
                company_data[col] = ""
        
        # Combinare text pentru clasificare
        combined_text = combine_company_text(company_data, weights)
        combined_texts.append(combined_text)
        
        # Adăugare date originale
        original_data = {col: row[col] for col in df.columns if col in row}
        company_data_list.append(original_data)
    
    return combined_texts, company_data_list 