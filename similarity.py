import torch
from transformers import BertModel, BertTokenizer
# Setting Device
if torch.backends.mps.is_built(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


def similarity(sentences, model=None, tokenizer=None):
    bertVersion = 'bert-large-uncased'
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(bertVersion)
    else:
        tokenizer = tokenizer
    if model is None:
        model = BertModel.from_pretrained(bertVersion)
    else:
        model = model

    encodings = tokenizer(sentences, return_tensors='pt', padding=True)
    encodings = encodings.to(device)

    # Create Embeddings
    model = model.to(device)
    with torch.no_grad():
        embeddings = model(**encodings)

    embeddings_cls = embeddings.last_hidden_state[:, 0, :]
    return torch.cosine_similarity(embeddings_cls[0].reshape(1,-1),
                                   embeddings_cls[1].reshape(1,-1)).item()

# print(similarity(['this is test sentence 1', 'random']))
