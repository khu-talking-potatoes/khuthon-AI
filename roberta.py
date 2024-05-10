from transformers import RobertaModel, RobertaTokenizer
import torch
from torch.nn import functional as F

def get_roberta_similarity(text1, text2):
    model_name='roberta-base'
    aggregate="mean"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    
    input_ids = [tokenizer.encode(t, add_special_tokens=True, return_tensors="pt") for t in [text1, text2]]
    
    with torch.no_grad():
        last_hidden_states = [model(i)[0] for i in input_ids]

    if aggregate == "cls":
        embeddings = [F.normalize(hs[0, 0], p=2, dim=0) for hs in last_hidden_states]
    elif aggregate == "mean":
        embeddings = [F.normalize(hs.mean(1).squeeze(0), p=2, dim=-1) for hs in last_hidden_states]

    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)

    return similarity