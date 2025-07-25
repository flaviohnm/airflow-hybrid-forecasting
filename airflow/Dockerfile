# 1. Use a imagem oficial do Airflow, que já tem o usuário 'airflow' configurado.
FROM apache/airflow:2.9.2-python3.11

# 2. Copia os arquivos de código e o requirements JÁ DEFININDO o dono correto.
COPY --chown=airflow:airflow ./src /opt/airflow/src
COPY --chown=airflow:airflow requirements.txt .

# 3. Instala as dependências diretamente no ambiente virtual do Airflow.
RUN pip install --no-cache-dir -r requirements.txt