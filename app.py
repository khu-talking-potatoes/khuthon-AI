from flask import Flask, request
from flask_restx import Api, Resource

from similarity import similarity

app = Flask(__name__)
api = Api(app)

@api.route('/sentence_length')
class SentenceLength(Resource):
    def post(self):
        # POST 요청에서 문장 두 개를 받음
        sentence1 = request.form.get('sentence1', '')  
        sentence2 = request.form.get('sentence2', '')
        
        sim = similarity([sentence1,sentence2])    
        
        # 길이 반환
        return {
            'similarity' : sim
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)