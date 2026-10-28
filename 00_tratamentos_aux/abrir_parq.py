import pandas as pd
import os

def abrir_parquet(caminho_arquivo: str) -> pd.DataFrame:
    """
    Abre um arquivo Parquet e retorna um DataFrame do pandas.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo Parquet.

    Retorna:
    pd.DataFrame: O DataFrame contendo os dados do arquivo Parquet.
    """
    df = pd.read_parquet(caminho_arquivo)
    return df       


if __name__ == "__main__":
    # Caminho para o arquivo Parquet
    
    ##  subid
    # caminho = os.path.expanduser("../_bases/sts.parquet")
    
    # SUB_ID
    caminho = os.path.expanduser("../_bases/test_subids.parquet")
    
    # Abrir o arquivo
    df = abrir_parquet(caminho)
    
    # Exibir informações básicas
    print("Primeiras linhas do DataFrame:")
    print(df.head())
    print("\nShape do DataFrame:", df.shape)
    print("\nColunas:", list(df.columns))