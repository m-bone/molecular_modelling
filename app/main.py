import os
from flask import Flask
from flask_restful import Resource, Api, reqparse
from eval_rf import evaluate_random_forest

app = Flask(__name__)
api = Api(app)

class RandomForest(Resource):
    def __init__(self) -> None:
        super().__init__()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('mol_str', type=str, required=True)
        
        args = parser.parse_args()
        mol_str = args['mol_str']

        return evaluate_random_forest(mol_str)

api.add_resource(RandomForest, '/rf')

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')