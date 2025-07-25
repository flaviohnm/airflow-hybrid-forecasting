# ===================================================================
# MÓDULO: experiment.py
# ===================================================================

import os
import pickle
from darts import TimeSeries
from darts.metrics import mape, mase

# Importações dos nossos módulos locais
from src.framework import BaseModel

def run_single_experiment(
    model: BaseModel,
    dataset_name: str,
    series: TimeSeries,
    forecast_horizon: int,
    results_path: str = 'results/'
):
    """
    Executa um único experimento de previsão: treina um modelo em um dataset,
    avalia o desempenho e salva as previsões e métricas.
    """
    
    # Cria os diretórios de resultados se não existirem
    pred_path = os.path.join(results_path, 'predictions')
    metrics_path = os.path.join(results_path, 'metrics')
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    print(f"--- Executando: Modelo '{model.name}' no Dataset '{dataset_name}' ---")

    try:
        # 1. Dividir a série em treino e teste
        train, test = series[:-forecast_horizon], series[-forecast_horizon:]

        # 2. Treinar o modelo
        model.fit(train, forecast_horizon)

        # 3. Realizar a previsão
        prediction = model.predict(forecast_horizon)

        # 4. Salvar a previsão em um arquivo CSV
        pred_df = prediction.pd_series().to_frame(name='prediction')
        pred_filename = os.path.join(pred_path, f"{dataset_name}_{model.name}.csv")
        pred_df.to_csv(pred_filename)

        # 5. Calcular as métricas de erro
        mape_score = mape(test, prediction)
        # O MASE precisa da série de treino para calcular a escala do erro
        mase_score = mase(test, prediction, train)

        metrics = {'MAPE': mape_score, 'MASE': mase_score}
        
        # 6. Salvar as métricas em um arquivo pickle
        metrics_filename = os.path.join(metrics_path, f"{dataset_name}_{model.name}.pkl")
        with open(metrics_filename, 'wb') as f:
            pickle.dump(metrics, f)
            
        print(f"Resultados para '{model.name}' em '{dataset_name}': MAPE={mape_score:.2f}%, MASE={mase_score:.3f}")

    except Exception as e:
        print(f"ERRO ao processar o modelo '{model.name}' no dataset '{dataset_name}': {e}")
        # A LINHA ABAIXO É CRUCIAL: ELA INFORMA AO AIRFLOW QUE A TAREFA FALHOU
        raise e