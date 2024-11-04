import pandas as pd
import polars as pl
import numpy as np
from faker import Faker
from .config import DEFAULT_LIBRARY

class DataGenerator:
    def __init__(self, library=DEFAULT_LIBRARY):
        self.library = library
        self.fake = Faker()

    def generate_column(self, num_rows, col_type):
        """
        Genera una colonna con dati di un tipo specifico.
        Args:
            num_rows (int): Numero di righe per la colonna.
            col_type (str): Tipo di dati della colonna ('numeric', 'string', 'datetime').

        Returns:
            list: Lista di dati casuali per la colonna specificata.
        """
        if col_type == 'numeric':
            return np.random.rand(num_rows) * 100  # Numeri casuali tra 0 e 100
        elif col_type == 'string':
            return [self.fake.name() for _ in range(num_rows)]
        elif col_type == 'datetime':
            return [self.fake.date_this_decade() for _ in range(num_rows)]
        else:
            raise ValueError("Tipo di colonna non valido.")

    def create_dataframe(self, num_rows, num_columns):
        """
        Crea un dataframe casuale di dimensioni specifiche.
        Args:
            num_rows (int): Numero di righe.
            num_columns (int): Numero di colonne.

        Returns:
            DataFrame: Un dataframe generato casualmente (Pandas o Polars).
        """
        data = {}
        for i in range(num_columns):
            col_type = i % 3  # Alterna tra 'numeric', 'string', 'datetime'
            if col_type == 0:
                data[f'num_col_{i}'] = self.generate_column(num_rows, 'numeric')
            elif col_type == 1:
                data[f'str_col_{i}'] = self.generate_column(num_rows, 'string')
            elif col_type == 2:
                data[f'date_col_{i}'] = self.generate_column(num_rows, 'datetime')

        # Genera il dataframe nella libreria scelta
        if self.library == 'pandas':
            return pd.DataFrame(data)
        elif self.library == 'polars':
            return pl.DataFrame(data)
        else:
            raise ValueError("Specifica una libreria valida: 'pandas' o 'polars'.")

    def generate_datasets(self, sizes):
        """
        Genera più dataframe per diverse dimensioni.
        Args:
            sizes (list): Lista di tuple con il numero di righe e colonne.
        
        Returns:
            dict: Dizionario di dataframe generati, con chiavi nel formato '{righe}x{colonne}'.
        """
        datasets = {}
        for num_rows, num_columns in sizes:
            df = self.create_dataframe(num_rows, num_columns)
            datasets[f"{num_rows}x{num_columns}"] = df
            print(f"Creato dataset {num_rows}x{num_columns} per {self.library}")
        return datasets