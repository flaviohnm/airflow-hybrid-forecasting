# ===================================================================
# MÓDULO: data_loader.py
# ===================================================================

import os
import pandas as pd
import statsmodels.api as sm

class DataLoader:
    """
    Classe responsável por carregar datasets de séries temporais clássicas.

    Os datasets são carregados da biblioteca statsmodels e salvos localmente
    em formato CSV para fácil acesso em execuções futuras.
    """
    def __init__(self, base_path='data/raw/'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def load_classic_ts_dataset(self, dataset_name: str) -> pd.Series:
        """
        Carrega um dataset específico ou o lê do cache local se já existir.

        Args:
            dataset_name (str): O nome do dataset a ser carregado.
                                Ex: 'AirPassengers', 'co2'.

        Returns:
            pd.Series: A série temporal carregada.
        """
        local_path = os.path.join(self.base_path, f"{dataset_name}.csv")
        
        if os.path.exists(local_path):
            return pd.read_csv(local_path, index_col=0, parse_dates=True).squeeze("columns")

        print(f"Dataset '{dataset_name}' não encontrado localmente. Carregando e salvando...")
        
        try:
            if dataset_name == 'AirPassengers':
                df = sm.datasets.get_rdataset("AirPassengers").data
                series = pd.Series(df['value'].values, index=pd.to_datetime(df['time'].apply(lambda x: f'{int(x)}-{((x % 1) * 12) + 1:.0f}')), name="AirPassengers")
                series.index.freq = 'MS'
            elif dataset_name == 'co2':
                data = sm.datasets.co2.load_pandas().data
                series = data['co2'].resample('W').mean().ffill().rename("CO2")
            elif dataset_name == 'nottem':
                # O rdataset "nottem" não tem um formato de data claro, criando um range.
                df = sm.datasets.get_rdataset("nottem").data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1920-01-01', periods=len(df), freq='AS-JAN'), name="NottinghamTemp")
            elif dataset_name == 'JohnsonJohnson':
                df = sm.datasets.get_rdataset("JohnsonJohnson").data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1960-01-01', periods=len(df), freq='QS-JAN'), name="JohnsonJohnson")
            elif dataset_name == 'UKgas':
                df = sm.datasets.get_rdataset("UKgas").data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1960-01-01', periods=len(df), freq='QS-JAN'), name="UKGas")
            elif dataset_name == 'Sunspots':
                df = sm.datasets.sunspots.load_pandas().data
                series = pd.Series(df['SUNACTIVITY'].values, index=pd.to_datetime(df['YEAR'], format='%Y'), name="Sunspots")
            elif dataset_name == 'Nile':
                df = sm.datasets.nile.load_pandas().data
                series = pd.Series(df['volume'].values, index=pd.to_datetime(df['year'], format='%Y'), name="Nile")
            elif dataset_name == 'ukdriverdeaths':
                # O rdataset "UKDriverDeaths" também requer ajuste de data.
                df = sm.datasets.get_rdataset("UKDriverDeaths").data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1969-01-01', periods=len(df), freq='MS'), name="UKDriverDeaths")
            else:
                raise ValueError(f"Dataset '{dataset_name}' não reconhecido.")
            
            series.to_csv(local_path)
            print(f"Dataset '{dataset_name}' salvo em '{local_path}'.")
            return series
            
        except Exception as e:
            print(f"Erro ao carregar o dataset '{dataset_name}': {e}")
            return None