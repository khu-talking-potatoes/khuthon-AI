from flask import Flask, request
from flask_restx import Api, Resource

from similarity import similarity
from rouge import rouge_score

app = Flask(__name__)
api = Api(app)

@api.route('/sentence_length')
class SentenceLength(Resource):
    def post(self):
        # POST 요청에서 문장 두 개를 받음
        sentence1 = request.form.get('sentence1', '')  
        sentence2 = request.form.get('sentence2', '')
        
        sim = similarity([sentence1,sentence2])    
        rouge = rouge_score([sentence1],[sentence2])
        len1 = len(sentence1)
        len2 = len(sentence2)

        return {
            'similarity' : sim,
            'rouge' : rouge,
            'len1' : len1,
            'len2' : len2
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)