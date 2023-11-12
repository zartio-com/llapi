from flask import Flask, jsonify, request
from wrappers import LlamaModel

app = Flask("llama-api")


@app.route('/generate', methods=['POST'])
def inference_predict():
    # start_time = time.time()
    data = request.json

    model_output = LlamaModel.predict(data['text'], data['settings'] if 'settings' in data else {})
    return jsonify({'text': model_output})
