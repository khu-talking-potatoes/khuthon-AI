from flask import Flask, request
from flask_restx import Api, Resource

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

app = Flask(__name__)
api = Api(app)

@api.route('/sentence_length')
class SentenceLength(Resource):
    def post(self):
        # POST 요청에서 문장 두 개를 받음
        sentence1 = request.form.get('sentence1', '')  
        sentence2 = request.form.get('sentence2', '')
        
        sim = similarity([sentence1,sentence2])    

        # 문장의 길이 계산
        #length1 = len(sentence1)
        #length2 = len(sentence2)
        
        # 길이 반환
        return {
            #'sentence1_length': length1,
            #'sentence2_length': length2,
            'similarity' : sim
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)