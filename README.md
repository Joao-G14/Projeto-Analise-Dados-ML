Modelo de Previsão de Churn com TensorFlow

Sobre o Projeto

Este projeto foi desenvolvido com o objetivo de aplicar conceitos de Machine Learning para resolver um problema de negócios focado na retenção de clientes
A proposta do projeto é construir uma Rede Neural capaz de identificar padrões de comportamento e prever a probabilidade de um cliente cancelar um serviço ou assinatura

Tecnologias Utilizadas


Python

Pandas

Numpy

Scikit-Learn

TensorFlow 

Keras


Estrutura do Projeto

BaseClientes.csv → Base de dados histórica contendo o perfil e comportamento dos usuários
previsao_churn.py → Script Python responsável pela construção treinamento e avaliação da Rede Neural

Tratamento e Preparação dos Dados

O processamento inicial dos dados foi realizado utilizando Pandas e Scikit-Learn
O script realiza:
Leitura e estruturação da base de dados de clientes
Separação entre variáveis preditivas e variável alvo
Divisão do conjunto de dados em amostras de treino e teste
Normalização das escalas matemáticas utilizando StandardScaler para otimizar o aprendizado da máquina

Arquitetura da Rede Neural

O modelo de Deep Learning foi desenvolvido utilizando a biblioteca TensorFlow
A estrutura contém:
Camada de entrada recebendo as características do cliente
Duas camadas ocultas densas com função de ativação ReLU para captura de relações não lineares
Camada de saída com função Sigmoid para retornar a probabilidade matemática do cancelamento
Otimizador Adam e cálculo de perda por entropia cruzada binária

Indicadores Desenvolvidos

O projeto avalia a performance através dos seguintes pontos:
Acurácia geral do modelo em dados desconhecidos
Probabilidade percentual de evasão de novos clientes inseridos no sistema
Classificação automática de risco para ações preventivas da equipe de retenção

Objetivo do Projeto

O projeto foi desenvolvido para prática de:
Desenvolvimento de algoritmos de Machine Learning e Deep Learning
Preparação de dados estruturados para modelagem preditiva
Criação e treinamento de Redes Neurais Artificiais
Integração de inteligência artificial com tomada de decisão em negócios

Autor:

João Gabriel Amaral
