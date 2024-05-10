from flask import Flask, request
from flask_restx import Api, Resource

app = Flask(__name__)
api = Api(app)

@api.route('/sentence_length')
class SentenceLength(Resource):
    def post(self):
        data = request.get_json()  # POST 요청으로부터 JSON 데이터 가져오기
        sentence1 = data.get('sentence1', '')  # 첫 번째 문장 가져오기
        sentence2 = data.get('sentence2', '')  # 두 번째 문장 가져오기
        
        length1 = len(sentence1)  # 첫 번째 문장의 길이 계산
        length2 = len(sentence2)  # 두 번째 문장의 길이 계산
        
        return {
            'sentence1_length': length1,
            'sentence2_length': length2
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)