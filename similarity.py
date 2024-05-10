import torch
from transformers import BertModel, BertTokenizer
# Setting Device
if torch.backends.mps.is_built(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


def similarity(sentences):
    bertVersion = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(bertVersion)
    model = BertModel.from_pretrained(bertVersion)

    encodings = tokenizer(sentences, return_tensors='pt', padding=True)
    encodings = encodings.to(device)

    # Create Embeddings
    model = model.to(device)
    with torch.no_grad():
        embeddings = model(**encodings)

    embeddings_avg = embeddings[0].mean(axis=2)

    return torch.cosine_similarity(embeddings_avg[0].reshape(1,-1), embeddings_avg[1].reshape(1,-1)).item()

# print(similarity(['this is test sentence 1', 'random']))