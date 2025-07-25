#!/bin/bash
# Este script gerencia o ambiente Docker do Airflow

echo "--- Parando e removendo contêineres antigos... ---"
docker-compose down --volumes --remove-orphans

echo "--- Construindo as imagens Docker... ---"
docker-compose build

echo "--- Iniciando os serviços do Airflow em segundo plano... ---"
docker-compose up -d

echo "--- Ambiente iniciado! Acesse http://localhost:8080 ---"