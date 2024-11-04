import pandas as pd
import polars as pl
import numpy as np
from faker import Faker
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # Per parallelizzare
from .config import DEFAULT_LIBRARY, NUM_SAMPLES

class DataGenerator:
    def __init__(self, library=DEFAULT_LIBRARY):
        self.library = library
        self.fake = Faker()

    def generate_column(self, num_rows, col_type):
        """Genera una colonna con dati di un tipo specifico."""
        if col_type == 'numeric':
            return np.random.rand(num_rows) * 100
        elif col_type == 'string':
            # Genera un batch di nomi una sola volta, poi replica fino a num_rows
            unique_names = [self.fake.name() for _ in range(min(num_rows, 1000))]
            return [unique_names[i % len(unique_names)] for i in range(num_rows)]
        elif col_type == 'datetime':
            start_date = np.datetime64('2010-01-01')
            end_date = np.datetime64('2020-12-31')
            return np.random.choice(np.arange(start_date, end_date, dtype='datetime64[D]'), num_rows)
        else:
            raise ValueError("Tipo di colonna non valido.")

    def create_dataframe(self, num_rows, num_columns):
        """Crea un dataframe casuale di dimensioni specifiche."""
        data = {}
        for i in range(num_columns):
            col_type = i % 3  # Alterna tra 'numeric', 'string', 'datetime'
            if col_type == 0:
                data[f'num_col_{i}'] = self.generate_column(num_rows, 'numeric')
            elif col_type == 1:
                data[f'str_col_{i}'] = self.generate_column(num_rows, 'string')
            elif col_type == 2:
                data[f'date_col_{i}'] = self.generate_column(num_rows, 'datetime')

        if self.library == 'pandas':
            return pd.DataFrame(data)
        elif self.library == 'polars':
            return pl.DataFrame(data)
        else:
            raise ValueError("Specifica una libreria valida: 'pandas' o 'polars'.")

    def generate_dataset_series(self, base_size, num_columns, num_samples=NUM_SAMPLES):
        """
        Genera una serie di dataset con numero di righe crescente.
        
        Args:
            base_size (int): Numero di righe di partenza per l'ordine di grandezza.
            num_columns (int): Numero di colonne.
            num_samples (int): Numero di dataset da generare per ogni ordine di grandezza.
        
        Returns:
            list: Lista di dataframe con numero di righe crescente.
        """
        datasets = []
        increment_step = int(base_size * 0.1)  # Incremento progressivo (10% del numero di righe base)

        for i in tqdm(range(num_samples), desc=f"Generating datasets for {base_size} rows, {num_columns} cols"):
            num_rows = base_size + (i * increment_step)
            df = self.create_dataframe(num_rows, num_columns)
            datasets.append((f"{num_rows}x{num_columns}", df))

        return datasets

    def generate_all_datasets(self, sizes):
        """
        Genera tutte le serie di dataset per ogni dimensione base in `sizes`.
        
        Args:
            sizes (list): Lista di tuple con numero di righe e colonne base.
        
        Returns:
            dict: Dizionario contenente le serie di dataset per ciascuna dimensione.
        """
        all_datasets = {}

        # Parallellizza la generazione dei dataset per ogni dimensione base
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate_dataset_series, base_size, num_columns)
                for base_size, num_columns in sizes
            ]
            for future, (base_size, num_columns) in zip(futures, sizes):
                all_datasets[f"{base_size}_rows_{num_columns}_cols"] = future.result()
        
        return all_datasets

    def save_datasets(self, datasets, output_dir="datasets"):
        """
        Salva i dataset generati in una cartella specifica con il nome {numero_righe}_{numero_colonne}.csv.
        
        Args:
            datasets (dict): Dizionario di serie di dataset generati.
            output_dir (str): Percorso della cartella di destinazione.
        """
        os.makedirs(output_dir, exist_ok=True)

        for base_key, dataset_series in datasets.items():
            for dataset_name, df in dataset_series:
                file_path = os.path.join(output_dir, f"{dataset_name}.csv")
                
                # Salva il dataframe come CSV a seconda della libreria
                if self.library == 'pandas':
                    df.to_csv(file_path, index=False)
                elif self.library == 'polars':
                    df.write_csv(file_path)
                
                print(f"Salvato dataset {dataset_name} in {file_path}")
