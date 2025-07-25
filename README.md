# Pipeline de Forecasting Híbrido com Airflow e Docker

Este projeto implementa uma pipeline de MLOps robusta para treinar, avaliar e comparar múltiplos modelos de forecasting de séries temporais, incluindo modelos estatísticos clássicos, redes neuronais e frameworks híbridos. A orquestração é feita com Apache Airflow e todo o ambiente é encapsulado em contentores Docker para garantir a reprodutibilidade.

## Principais Características

- **Orquestração de Pipeline:** Utilização do Apache Airflow com o TaskFlow API para definir a pipeline de forma declarativa em Python.
- **Ambiente Reprodutível:** Todo o ambiente, incluindo dependências Python e serviços, é gerido pelo Docker e Docker Compose.
- **Execução Paralela e Escalável:** Uso do CeleryExecutor do Airflow para permitir a execução distribuída de tarefas.
- **Gestão de Recursos:** Implementação de Pools no Airflow para gerir a concorrência de tarefas pesadas (treino de modelos de redes neuronais) e leves (modelos estatísticos), otimizando o uso de recursos.
- **Modelos Implementados:**
  - Baselines Estatísticos: `PureARIMA` (via `auto_arima`).
  - Modelos de Redes Neuronais: `PureLSTM`, `PureNBEATS`.
  - Frameworks Híbridos: `ARIMA-LSTM`, `ARIMA-MLP`, e variações com estratégias `direct` e `mimo`.
- **Relatório Automatizado:** Geração automática de um relatório completo em Markdown ao final da pipeline, incluindo tabelas de métricas (MAPE, MASE), análise estatística (Teste de Friedman, Diagrama de Diferença Crítica) e gráficos de comparação de previsões.

## Estrutura do Projeto

```text
.
├── dags/
│   └── ml_pipeline_dag.py
├── data/
│   └── raw/
├── logs/
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
├── src/
│   ├── data_loader.py
│   ├── experiment.py
│   ├── framework.py
│   ├── models.py
│   └── reporting.py
├── .gitignore
├── docker-compose.yaml
├── Dockerfile
├── entrypoint.sh
└── requirements.txt

## Configuração e Instalação

**Pré-requisitos:**
- Git
- Docker Desktop com WSL2 configurado
- `pyenv` para gestão de versões Python (opcional, mas recomendado)

**Passos:**

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/](https://github.com/)[SEU-USUARIO]/[SEU-REPOSITORIO].git
    cd [SEU-REPOSITORIO]
    ```

2.  **Configure o Ambiente Virtual (Recomendado):**
    ```bash
    # Defina a versão do Python para o projeto
    pyenv local 3.11.9
    
    # Crie e ative o ambiente virtual
    python -m venv venv
    source venv/Scripts/activate
    
    # Instale as dependências
    pip install -r requirements.txt
    ```

3.  **Inicie o Ambiente Airflow com Docker:**
    *Execute o script de inicialização. Na primeira vez, ele irá construir a imagem Docker, o que pode demorar alguns minutos.*
    ```bash
    ./entrypoint.sh
    ```

4.  **Acesse a Interface do Airflow:**
    * Abra o seu navegador e aceda a `http://localhost:8080`.
    * Use o login `airflow` e a senha `airflow`.

## Como Usar

1.  Na interface do Airflow, ative a DAG `hybrid_forecasting_pipeline`.
2.  Dispare uma execução manual clicando no botão "Play".
3.  Acompanhe a execução na vista "Grid". As tarefas serão executadas sequencialmente ou em paralelo, de acordo com as regras dos pools.
4.  Ao final da execução, o relatório completo estará disponível em `results/reports/final_report.md`.

## Conceitos de Avaliação Implementados

Este projeto segue as boas práticas recomendadas no artigo "Forecast evaluation for data scientists: common pitfalls and best practices", incluindo:
- Uso de métricas escaladas como **MASE**.
- Análise estatística robusta com **Teste de Friedman** para comparar múltiplos modelos em múltiplos datasets.
- Geração de **Diagramas de Diferença Crítica** para visualizar os resultados dos testes estatísticos.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o ficheiro `LICENSE` para mais detalhes.