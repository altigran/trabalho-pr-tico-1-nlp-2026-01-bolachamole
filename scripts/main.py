from Modelo import Modelo
from metricas import jaccard, perplexity, self_bleu, distinct_n
import json
import pandas as pd

def main():
    modelos = ["deepseek-ai/deepseek-v4-pro", "z-ai/glm-5.1", "meta/llama-4-maverick-17b-128e-instruct"]

    # lê as mensagens
    with open("data/prompts.json", 'r') as p:
        prompts = json.load(p)
        mensagens = prompts.get('2')
    tipos = prompts.get('1').keys()
    jac = {k: [] for k in tipos}
    perp = {k: [] for k in tipos}
    sb = {k: [] for k in tipos}
    dst1 = {k: [] for k in tipos}
    dst2 = {k: [] for k in tipos}

    # testa as configurações em cada modelo
    for m in modelos:
        print("Testando modelo", m)
        llm = Modelo(m)
        for k,v in prompts.get('1').items():
            prompt = f"{v}{mensagens}"
            resposta = llm.respondePrompt(prompt)
            if (resposta):
                if (resposta.startswith("```json")):
                    resposta = resposta.replace("```json", '')
                if (resposta.endswith("```")):
                    resposta = resposta.replace("```", '')
                print("\nRespostas obtida com sucesso para", k)
                nome_arq = f"results/{m.replace('/','_')}_{k}.json"
                with open(nome_arq, 'w') as arq:
                    arq.write(resposta)
                # roda as métricas automáticas
                resposta = json.loads(resposta)
                if (k == "base"):
                    resps = list(resposta.values())
                else:
                    resps = []
                    for i in range(1,21):
                        for j in range(1,4):
                            resps.append(resposta.get(str(i)).get(str(j))[0])
                jac[k].append(jaccard(resps))
                perp[k].append(perplexity(resps))
                sb[k].append(self_bleu(resps))
                dst1[k].append(distinct_n(resps, 1))
                dst2[k].append(distinct_n(resps, 2))

    # faz uma tabela para comparar as métricas
    d = {
        "Jaccard (Baseline)": jac.get("base"), 
        "Jaccard (VS simples)": jac.get("vs1"), 
        "Jaccard (VS agressivo)": jac.get("vs2"),
        "Perplexity (Baseline)": perp.get("base"),
        "Perplexity (VS simples)": perp.get("vs1"),
        "Perplexity (VS agressivo)": perp.get("vs2"),
        "Self-Bleu (Baseline)": sb.get("base"),
        "Self-Bleu (VS simples)": sb.get("vs1"),
        "Self-Bleu (VS agressivo)": sb.get("vs2"),
        "Distinct-1 (Baseline)": dst1.get("base"),
        "Distinct-1 (VS simples)": dst1.get("vs1"),
        "Distinct-1 (VS agressivo)": dst1.get("vs2"),
        "Distinct-2 (Baseline)": dst2.get("base"),
        "Distinct-2 (VS simples)": dst2.get("vs1"),
        "Distinct-2 (VS agressivo)": dst2.get("vs2")
    }
    tabela = pd.DataFrame(data=d, index=modelos)
    tabela.to_csv("results/comparacao.csv")
    print("Tabela compartiva exportada em results/comparacao.csv")

if __name__=="__main__":
    main()
