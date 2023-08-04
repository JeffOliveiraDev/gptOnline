from flask import Flask, render_template, request
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

app = Flask(__name__)

# Configurar o modelo e o índice aqui
# Certifique-se de carregar o modelo e criar o índice fora do contexto da função de rota para evitar recarregar a cada solicitação.

loader = TextLoader("./data/data.txt")
index_instance = VectorstoreIndexCreator().from_loaders([loader])
model = ChatOpenAI()

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']

        # Aqui, você usa a variável 'query' para fazer a solicitação ao modelo e obter a resposta.

        # Primeiro, obtemos o resultado da consulta usando o modelo e o índice.
        response = index_instance.query(query, llm=model)

        # Observe que a variável 'response' contém uma lista de resultados da consulta.
        # Você pode querer processar essa lista e extrair a resposta mais relevante de acordo com suas necessidades.

        # Por exemplo, supondo que a resposta mais relevante esteja no primeiro item da lista:
        if response:
            answer = response
        else:
            answer = "Desculpe, não encontrei uma resposta para sua requisição."

        return render_template('index.html', query=query, response=answer)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
