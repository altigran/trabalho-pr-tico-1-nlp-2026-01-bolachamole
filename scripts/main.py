from Modelo import Modelo
from dotenv import load_dotenv
import os
import csv

def main():
    load_dotenv()
    modelos = [""]

    for m in modelos:
        print("Testando modelo", m)
        llm = Modelo(os.environ.get("API_KEY"), m)
        # lê as mensagens
        with open("data/mensagens.txt", 'r') as p:
            for msg in p:
                resposta = llm.respondePrompt(msg.strip())
                if (resposta):

if __name__=="__main__":
    main()
