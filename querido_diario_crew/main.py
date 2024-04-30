import os
from dotenv import load_dotenv
load_dotenv()

from crew import QueridoDiarioCrew

def run():
    inputs = {
        "claim": 'A prefeitura de São José dos Campos investiu 17.350.000,00 em coleta seletiva.',
        "context": {
            "published_since": "1 de Janeiro de 2022",
            "published_until": "30 de Dezembro de 2022",
            "city": "São José dos Campos"
        }
    }
    QueridoDiarioCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()