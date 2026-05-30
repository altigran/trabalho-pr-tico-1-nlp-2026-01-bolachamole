from openai import OpenAI
import os

class Modelo:
    def __init__(self, nome_modelo):
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.nome = nome_modelo

    def respondePrompt(self, prompt):
        try:
            resposta = self.client.chat.completions.create(
                model=self.nome,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return resposta.choices[0].message.content
        except Exception as erro:
            print(erro)
        return None
