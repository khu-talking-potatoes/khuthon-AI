from flask import Flask, request
from flask_restx import Api, Resource
from flask_cors import CORS

from similarity import similarity
from rouge import rouge_score
from bleu import bleu_score
from mauve import mauve_score
from bertscore import bert_score
from roberta import get_roberta_similarity

### Bert Model Loading
import torch
from transformers import BertModel, BertTokenizer
# Setting Device
if torch.backends.mps.is_built(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')
###

app = Flask(__name__)
CORS(app)  # 모든 출처에서의 요청을 허용합니다.
api = Api(app)


@api.route('/sentence_length')
class SentenceLength(Resource):
    def post(self):
        # POST 요청에서 문장 두 개를 받음
        data = request.get_json()  # POST 요청으로부터 JSON 데이터 가져오기
        sentences = data.get('data', [])
  
        answer1 = sentences[0].get('answer', '')  # 모델의 답변
        answer2 = sentences[1].get('answer', '')  # 모델의 답변

        sentence1 = str(answer1)
        sentence2 = str(answer2)

        sim = similarity([sentence1, sentence2], model, tokenizer)
        precision, recall, rouge, f1_score = rouge_score([sentence1], [sentence2])
        bleu = bleu_score([sentence1], [sentence2])
        mauve = mauve_score([sentence1], [sentence2])
        bertscore = bert_score([sentence1], [sentence2])
        # roberta = get_roberta_similarity(sentence1, sentence2)
        len1 = len(sentence1)
        len2 = len(sentence2)

        return {
            'similarity': sim,
            'rouge': rouge,
            'rouge_precision':precision,
            'rouge_recall':recall,
            'rouge_fscore':f1_score,
            'bleu':bleu,
            'mauve':mauve,
            'bertscore_f1':bertscore,
            # 'roberta': roberta,
            'len1': len1,
            'len2': len2
        }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
