# ===================================================================
# MÓDULO: reporting.py
# ===================================================================

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import autorank
from typing import List


def load_all_metrics(metrics_path: str, model_names: List[str], dataset_names: List[str]) -> pd.DataFrame:
    all_metrics = []
    for ds_name in dataset_names:
        row = {'Dataset': ds_name}
        for model_name in model_names:
            file_path = os.path.join(metrics_path, f"{ds_name}_{model_name}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    metrics = pickle.load(f)
                    row[model_name] = metrics
        all_metrics.append(row)
    df_metrics = pd.DataFrame(all_metrics).set_index('Dataset')
    return df_metrics

def generate_metric_tables(df_metrics: pd.DataFrame) -> (str, str, pd.DataFrame):
    df_mape = df_metrics.map(lambda x: x.get('MAPE') if isinstance(x, dict) else None).dropna(axis=1, how='all')
    df_mase = df_metrics.map(lambda x: x.get('MASE') if isinstance(x, dict) else None).dropna(axis=1, how='all')
    mape_table_md = df_mape.to_markdown(floatfmt=".2f")
    mase_table_md = df_mase.to_markdown(floatfmt=".3f")
    return mape_table_md, mase_table_md, df_mape

def generate_pd_chart(df_mape: pd.DataFrame, images_path: str): # ALTERADO: Recebe o caminho das imagens
    mean_mapes = df_mape.mean()
    best_model_name = mean_mapes.idxmin()
    best_model_mape = mean_mapes.min()
    pd_results = {
        model_name: 100 * (mape_a - best_model_mape) / mape_a
        for model_name, mape_a in mean_mapes.items() if model_name != best_model_name
    }
    df_pd = pd.DataFrame.from_dict(pd_results, orient='index', columns=['PD(%)']).sort_values(by='PD(%)')
    
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x=df_pd.index, y=df_pd['PD(%)'], palette='viridis', hue=df_pd.index, legend=False)
    plt.title(f'Ganho Percentual (MAPE) em Relação ao Melhor Modelo ({best_model_name})', fontsize=16)
    plt.ylabel('Ganho (%)', fontsize=12)
    plt.xlabel('Modelo de Comparação', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    chart_path = os.path.join(images_path, 'pd_chart.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de Ganho Percentual salvo em: {chart_path}")

def generate_forecast_plots(model_names: List[str], dataset_names: List[str], results_path: str, images_path: str, forecast_horizon: int): # ALTERADO: Recebe o caminho das imagens
    predictions_path = os.path.join(results_path, 'predictions')
    for ds_name in dataset_names:
        try:
            raw_data_path = f"/opt/airflow/data/raw/{ds_name}.csv"
            full_series_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
            test_series = full_series_df[-forecast_horizon:].squeeze().rename('Valores Reais')

            plt.figure(figsize=(14, 7))
            plt.plot(test_series.index, test_series.values, label='Valores Reais', color='black', linewidth=2.5, zorder=10)

            for model_name in model_names:
                file_path = os.path.join(predictions_path, f"{ds_name}_{model_name}.csv")
                if os.path.exists(file_path):
                    pred_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    plt.plot(pred_df.index, pred_df['prediction'], label=f'Previsão {model_name}', linestyle='--')

            plt.title(f'Comparação de Previsões para o Dataset {ds_name}', fontsize=16)
            plt.xlabel('Data', fontsize=12)
            plt.ylabel('Valores', fontsize=12)
            plt.legend()
            plot_path = os.path.join(images_path, f'forecast_plot_{ds_name}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Gráfico de previsão para '{ds_name}' salvo em: {plot_path}")
        except FileNotFoundError:
            print(f"Arquivo de dados brutos '{raw_data_path}' não encontrado. Pulando gráfico de previsão.")
        except Exception as e:
            print(f"Erro ao gerar gráfico para '{ds_name}': {e}")

def run_autorank_analysis(df_metrics: pd.DataFrame, images_path: str) -> str: # ALTERADO: Recebe o caminho das imagens
    print("Executando análise estatística com autorank...")
    try:
        result = autorank.autorank(df_metrics, alpha=0.05, verbose=False)
        autorank.plot_stats(result, allow_insignificant=True)
        cd_diagram_path = os.path.join(images_path, 'cd_diagram.png')
        plt.savefig(cd_diagram_path, bbox_inches='tight')
        plt.close()
        print(f"Diagrama de Diferença Crítica salvo em: {cd_diagram_path}")
        
        summary = f"**Teste Omnibus:** {result.omnibus} (p-valor: {result.pvalue:.4f})\n\n"
        summary += f"**Teste Post-hoc:** {result.posthoc}\n\n"
        if result.pvalue < 0.05:
            summary += "Conclusão: Existem diferenças estatisticamente significativas entre os modelos.\n\n"
        else:
            summary += "Conclusão: Não há evidência de diferenças estatisticamente significativas entre os modelos.\n\n"
        
        summary += "**Tabela de Rankings:**\n"
        summary += result.rankdf.to_markdown(floatfmt=".2f")
        
        return summary
        
    except Exception as e:
        print(f"Erro ao executar o autorank: {e}")
        return "A análise estatística com autorank falhou."

# --- Função Principal ---

def generate_final_report(
    model_names: List[str],
    dataset_names: List[str],
    results_path: str = '/opt/airflow/results/',
    forecast_horizon: int = 10
):
    reports_path = os.path.join(results_path, 'reports')
    metrics_path = os.path.join(results_path, 'metrics')
    images_path = os.path.join(reports_path, 'images')
    os.makedirs(images_path, exist_ok=True)
    
    print("\n--- Iniciando Geração do Relatório Final Aprimorado ---")
    
    df_metrics_raw = load_all_metrics(metrics_path, model_names, dataset_names)
    if df_metrics_raw.empty:
        print("Nenhum resultado de métrica encontrado.")
        return

    mape_table_md, mase_table_md, df_mape = generate_metric_tables(df_metrics_raw)
    
    autorank_summary = "Análise de MAPE não pôde ser executada por falta de dados."
    if not df_mape.empty:
        generate_pd_chart(df_mape, images_path)
        autorank_summary = run_autorank_analysis(df_mape, images_path)

    generate_forecast_plots(model_names, dataset_names, results_path, images_path, forecast_horizon)

    # ALTERADO: Atualiza os caminhos das imagens no relatório Markdown
    report_content = f"""
# Relatório de Análise Comparativa de Modelos
Este relatório apresenta os resultados dos experimentos de previsão.

## Tabelas de Métricas de Erro
### Mean Absolute Percentage Error (MAPE)
Valores menores indicam melhor desempenho.
{mape_table_md}

### Mean Absolute Scaled Error (MASE)
Valores menores que 1 indicam que o modelo é melhor que uma previsão naíve da sazonalidade anterior.
{mase_table_md}

## Análise de Desempenho Relativo
### Ganho Percentual de Desempenho (vs. Melhor Modelo)
O gráfico mostra o quão melhor (em %) cada modelo foi em relação ao melhor modelo geral em termos de MAPE médio.
![Gráfico de Ganho Percentual](images/pd_chart.png)

## Análise Estatística (AutoRank)
O Diagrama de Diferença Crítica (CD Diagram) resume visualmente os resultados do teste de Friedman e Nemenyi. Modelos conectados por uma linha horizontal não possuem uma diferença estatisticamente significativa em seus rankings de desempenho.
![Diagrama de Diferença Crítica](images/cd_diagram.png)

### Resumo do AutoRank
{autorank_summary}

## Gráficos de Previsão por Dataset
"""
    
    for ds_name in dataset_names:
        plot_filename = f'forecast_plot_{ds_name}.png'
        if os.path.exists(os.path.join(images_path, plot_filename)):
            report_content += f"\n### {ds_name}\n![Previsões para {ds_name}](images/{plot_filename})\n"

    # ALTERADO: O caminho do relatório agora está correto na raiz da pasta de relatórios
    report_file_path = os.path.join(reports_path, 'final_report.md')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"\n--- Relatório Final Aprimorado gerado com sucesso em: {report_file_path} ---")