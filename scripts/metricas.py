from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def tokeniza(texto):
    texto_sem_pontuacao = ' '.join([p for p in texto if p not in ['.',',','!','?','"']])
    return texto_sem_pontuacao.lower().split()

def jaccard(respostas):
    texto1 = respostas[0]
    a = set(tokeniza(texto1))
    c = a
    for i in range(1, len(respostas)):
        b = set((respostas[i]))
        a = a.intersection(b)
        c = c.union(b)
    if(len(c) != 0):
        return len(a) / len(c)
    return 0

def perplexity(respostas, modelo="gpt2"):
    texto = ' '.join(respostas)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(modelo).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(texto, return_tensors="pt").to(model.device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
   
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return torch.exp(outputs.loss).item()

def self_bleu(respostas):
    score = []
    for i in range(len(respostas)):
        hipotese = tokeniza(respostas[i])
        referencias = [tokeniza(r) for r in respostas if r != respostas[i]]
        chencherry = SmoothingFunction()
        score.append(sentence_bleu(referencias, hipotese, smoothing_function=chencherry.method1))
    if (len(score) != 0):
        return sum(score) / len(score)
    return 0

def distinct_n(respostas, n):
    texto = ' '.join(respostas)
    todos_ngramas = list(ngrams(tokeniza(texto), n))
    if (len(todos_ngramas) != 0): 
        ngramas_unicos = set(todos_ngramas)
        return len(ngramas_unicos) / len(todos_ngramas)
    return 0
