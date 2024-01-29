import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '../PKGMethod'))
import requests
from flask_cors import *
from flask import request
from flask import jsonify
from flask import Flask, Response
from PKGMethod.Extraction.Zephyr7B import NER, RE, EE, IE


app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/Zephyr7B/IE', methods=['POST'])
@cross_origin()
def getIeResult():
    json_data = request.get_json()
    text = json_data['text']

    result = IE(text)

    return jsonify({
        'text': text,
        'result': result
    })


def main():
    app.run(host='0.0.0.0', port=4003, debug=False)


if __name__ == '__main__':
    main()
