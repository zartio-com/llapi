import argparse
import time
from api import app
from loaders.loader_exllamav2 import ExLlamaV2ModelWrapper
from wrappers import LlamaModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model directory", required=True)
    parser.add_argument("--listen", help="Listen on all network adapters", action='store_true')
    parser.add_argument("--port", help="Listening port - default 5000", type=int, default=5000)
    args = parser.parse_args()

    start_time = time.time()
    LlamaModel.load_model(ExLlamaV2ModelWrapper, f'_data/models/{args.model}')
    print(f"Model loaded in {time.time() - start_time} seconds")

    app.run(host='0.0.0.0' if args.listen else None, port=args.port)
