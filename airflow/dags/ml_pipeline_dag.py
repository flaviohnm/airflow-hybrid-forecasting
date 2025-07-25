from __future__ import annotations
import pendulum
from airflow.decorators import dag, task
from itertools import product
from datetime import timedelta
import sys

# CORREÇÃO: Adiciona a pasta /opt/airflow ao caminho do Python
sys.path.append('/opt/airflow')

LISTA_DE_DATASETS = ['AirPassengers', 'Nile', 'UKgas', 'Sunspots', 'ukdriverdeaths']
FORECAST_HORIZON = 10
N_EPOCHS_FOR_DEV = 100
BATCH_SIZE_FOR_DEV = 16

# Nomes dos modelos que devem rodar sozinhos no pool dedicado
HEAVY_MODELS = ["HySF_Direct", "HySF_MIMO", "ARIMA-LSTM", "PureLSTM", "PureNBEATS"]

default_args = {
    'owner': 'airflow', 'execution_timeout': timedelta(hours=2),
    'retries': 3, 'retry_delay': timedelta(minutes=10),
}

@dag(
    dag_id="hybrid_forecasting_pipeline",
    start_date=pendulum.datetime(2024, 1, 1, tz="America/Recife"),
    schedule=None, catchup=False, default_args=default_args,
    tags=["ml", "forecasting", "final_project"],
)
def hybrid_forecasting_dag():

    MODEL_CONFIGS = [
        {'model_class_str': 'HybridForecastingFramework', 'name': "HySF_Direct", 'params': {
            'non_linear_model_class_str': 'NBEATSModel', 'strategy': 'direct',
            'non_linear_model_params': {'input_chunk_length': 24, 'n_epochs': N_EPOCHS_FOR_DEV, 'random_state': 42, 'batch_size': BATCH_SIZE_FOR_DEV}
        }},
        {'model_class_str': 'HybridForecastingFramework', 'name': "HySF_MIMO", 'params': {
            'non_linear_model_class_str': 'NBEATSModel', 'strategy': 'mimo',
            'non_linear_model_params': {'input_chunk_length': 24, 'n_epochs': N_EPOCHS_FOR_DEV, 'random_state': 42, 'batch_size': BATCH_SIZE_FOR_DEV}
        }},
        {'model_class_str': 'HybridForecastingFramework', 'name': "ARIMA-LSTM", 'params': {
            'non_linear_model_class_str': 'RNNModel', 'strategy': 'recursive',
            'non_linear_model_params': {'model': 'LSTM', 'input_chunk_length': 24, 'n_epochs': N_EPOCHS_FOR_DEV, 'random_state': 42, 'batch_size': BATCH_SIZE_FOR_DEV}
        }},
        {'model_class_str': 'HybridForecastingFramework', 'name': "ARIMA-MLP", 'params': {
            'non_linear_model_class_str': 'MLPRegressorWrapper', 'strategy': 'mimo',
            'non_linear_model_params': {'n_lags': 24, 'output_chunk_length': FORECAST_HORIZON, 'max_iter': 1000}
        }},
        {'model_class_str': 'PureARIMA', 'name': "PureARIMA", 'params': {}},
        {'model_class_str': 'PureLSTM', 'name': "PureLSTM", 'params': {'n_lags': 24, 'n_epochs': N_EPOCHS_FOR_DEV, 'batch_size': BATCH_SIZE_FOR_DEV}},
        {'model_class_str': 'PureNBEATS', 'name': "PureNBEATS", 'params': {'n_lags': 24, 'n_epochs': N_EPOCHS_FOR_DEV, 'batch_size': BATCH_SIZE_FOR_DEV}},
    ]

    @task
    def get_light_configs() -> list:
        """Retorna uma lista apenas com as configurações de modelos leves."""
        light_models = [mc for mc in MODEL_CONFIGS if mc['name'] not in HEAVY_MODELS]
        return [{'model_config': mc, 'dataset_name': dn} for mc, dn in product(light_models, LISTA_DE_DATASETS)]

    @task
    def get_heavy_configs() -> list:
        """Retorna uma lista apenas com as configurações de modelos pesados."""
        heavy_models = [mc for mc in MODEL_CONFIGS if mc['name'] in HEAVY_MODELS]
        return [{'model_config': mc, 'dataset_name': dn} for mc, dn in product(heavy_models, LISTA_DE_DATASETS)]

    def _run_single_experiment_logic(config: dict):
        """Função auxiliar com a lógica de execução para evitar repetição de código."""
        from src.data_loader import DataLoader
        from src.framework import HybridForecastingFramework, MLPRegressorWrapper
        from src.models import PureARIMA, PureLSTM, PureNBEATS
        from src.experiment import run_single_experiment
        from darts import TimeSeries
        from darts.dataprocessing.transformers import MissingValuesFiller
        from darts.models import NBEATSModel, RNNModel

        model_config = config['model_config']
        dataset_name = config['dataset_name']
        
        MODEL_CLASS_MAP = {
            'NBEATSModel': NBEATSModel, 'RNNModel': RNNModel,
            'MLPRegressorWrapper': MLPRegressorWrapper, 'PureARIMA': PureARIMA,
            'PureLSTM': PureLSTM, 'PureNBEATS': PureNBEATS,
            'HybridForecastingFramework': HybridForecastingFramework
        }
        
        model_name = model_config['name']
        model_params = model_config['params'].copy()
        ModelClass = MODEL_CLASS_MAP[model_config['model_class_str']]

        if model_config['model_class_str'] == 'HybridForecastingFramework':
            class_str = model_params.pop('non_linear_model_class_str')
            model_params['non_linear_model_class'] = MODEL_CLASS_MAP[class_str]

        model = ModelClass(name=model_name, **model_params)

        data_loader = DataLoader()
        pd_series = data_loader.load_classic_ts_dataset(dataset_name)
        filler = MissingValuesFiller()
        darts_series = filler.transform(TimeSeries.from_series(pd_series, fill_missing_dates=True, freq=None))
        
        run_single_experiment(model, dataset_name, darts_series, FORECAST_HORIZON)

    @task(pool="heavy_task_pool")
    def run_heavy_experiment_task(config: dict):
        _run_single_experiment_logic(config)

    @task
    def run_light_experiment_task(config: dict):
        _run_single_experiment_logic(config)

    @task
    def generate_report_task(light_results, heavy_results):
        from src.reporting import generate_final_report
        model_names = sorted(list(set(item['name'] for item in MODEL_CONFIGS)))
        generate_final_report(model_names, LISTA_DE_DATASETS)

    # Orquestração
    light_configs = get_light_configs()
    heavy_configs = get_heavy_configs()
    
    light_runs = run_light_experiment_task.expand(config=light_configs)
    heavy_runs = run_heavy_experiment_task.expand(config=heavy_configs)
    
    generate_report_task(light_results=light_runs, heavy_results=heavy_runs)

hybrid_forecasting_dag()