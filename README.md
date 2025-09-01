# Chatbot_Vendas

## 1. Objetivo:
Este projeto foi proposto pela empresa Sesi/Senai, com o objetivo de desenvolvimento de um chatbot que respondesse perguntas sobre dados de vendas, com base em três bases de dados (produtos, vendas e vendedores) e que também realizasse previsões de vendas futuras.

## 2. Ferramentas:
Python, Streamlit e OpenAI

## 3. Biblioteca para instalar:
'streamlit', 'pandas', 'openai', 'scikit-learn', 'xgboost', 'matplotlib', 'numpy'

## 4. Arquivos necessários:
Código do Chatbot completo:
 - app.py
Ícone da página web:
 - sesi-logo.png
Base de dados de produtos:
 - base_produtos.xlsx
Base de dados de vendas:
 - base_vendas.xlsx
Base de dados de vendedores:
 - base_vendedores.xlsx

## 5. Passo a passo:
### 1° Passo - Preparar ambiente local
O primeiro requisito básico para rodar o código em ambiente local, é ter a linguagem de programação Python instalada no computador. Outro requisito é o usuário possuir em sua máquina alguma IDE de leitura e execução de códigos Python (documentos no formato ".py"), no caso do Chatbot foi utilizado o Visual Studio Code (VSCode), que também necessita da extensão Python dentro dele para interpretar a linguagem de programação.
### 2° Passo - Instalação de pacotes Python
O documento requeriments.txt possui uma lista de pacotes utilizados no código app.py (do Chatbot), sendo necessário a instalação de todos estes pacotes para o funcionamento do Chatbot em ambiente local. Para que estas instalações sejam feitas é necessario abrir o Prompt de Comando do computador (também pode ser feito via terminal do VSCode) e dentro dele digitar o comando "pip install" juntamente com o nome dos pacotes Python que deseja instalar, em seguida pressionar a tecla "Enter" do teclado. Outra opção mais prática para instalação de todos os pacotes de uma vez, é digitando no terminal do VSCode o comando "pip install -r requeriments.txt" que funcionara apenas se o usuário possuir o arquivo requeriments.txt baixado na mesma pasta em que o terminal está configurado.
