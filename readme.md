# LlAPI

LlAPI is a simple, lightweight Llama inference API.
Currently only ExllamaV2 is supported.

Speed increase over text-generation-webui is about 50% (on my machine).
Tested with Trurl-2-7B-GPTQ 4bit 128g on RTX 4090 (Windows 11).
Exact numbers: 40-60 tokens/s with oobabooga's text-generation-webui, steady 90+ tokens/s with LlAPI.

Nothing fancy so far is going on here, just a simple Flask API with a single endpoint.

Soon this will be the backend API for [Krzyś](https://github.com/zartio-com/krzys)

## Installation

Requirements: Python 3.10 & Windows
(for now, it is possible to replace wheels for linux and run 
this, but for now i'm providing only Windows installation)

1. `pip -m venv venv`
2. `./venv/Scriptis`
3. `pip install -r requirements.txt`

## Running the API

`python main.py --model <model_directory_name>`

## Usage

Put your models inside _data/models directory.
For `--model` option just provide the directory name,
not the full path.

To listen on all interfaces, use `--listen`

## Calling the API

`curl -X POST -H "Content-Type: application/json" -d '{"text": "Jack Daniels is"}' http://localhost:5000/generate`

Available parameters for generation:

```json
{
    "text": "Jack Daniels is",
    "settings": {
      "tokens_limit": 50,
      "seed": 1,
      "sampler": {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
        "typical": 0,
        "repetition_penalty": 1.0,
        "disallow_eos_token": true
      }
    }
}
```

Response format:

```json
{
  "text": "generated text"
}
```