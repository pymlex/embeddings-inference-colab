# Embeddings Inference Server

A small FastAPI service that exposes sentence-transformers and Hugging Face embedding models through a lightweight HTTP API, including an OpenAI-compatible `/v1/embeddings endpoint`. The project is runnable on Colab and can be used with Langchain tools or frameworks like Flowise.

## Functional abilities

* Serve embeddings from any Hugging Face / sentence-transformers model.
* Two API shapes:
  * `/` — accepts `input`, `inputs`, list of strings or raw string and returns raw embeddings.
  * `/v1/embeddings` — OpenAI-compatible response when `model` key is supplied, otherwise returns raw embeddings.
* Optional public exposure via ngrok.
* Designed for running from the included Colab notebook.

## Repo base structure

```
├── app/
│   ├── main.py               
│   └── model.py               
├── embeddings_inference_server.ipynb
├── run_server.py             
├── Makefile                  
└── requirements.txt
````

---

## Run in Colab

Open `embeddings_inference_server.ipynb` in Google Colab and run the cells. The notebook installs `requirements.txt`, sets environment variables in the session and runs `run_server.py`. When `NGROK_AUTH_TOKEN` is provided the notebook prints a public ngrok URL. Environment variables are:

* `MODEL_DIR` — HF / sentence-transformers model id or local path.
* `NGROK_AUTH_TOKEN` — optional ngrok token.

---

## Example test

Run the command:

````
curl -X POST https://your-ngrok-server/ -H "Content-Type: application/json" -d "{\"inputs\":[\"test\"]}"
````

Expected result:

````
[[0.03982217609882355,-0.030197851359844208,-0.12821190059185028, ...
````

---

## API

### POST `/`

Accepts payloads in one of these shapes:

* `{"input": "single text"}`
* `{"inputs": ["text1", "text2"]}`
* 
Response is a JSON array of embeddings. Each embedding is an array of floats.

Example request:

```json
{"inputs":["This is a test","Another sentence"]}
```

Example response:

```json
[
  [0.00123, -0.0234, ...],
  [0.00345, -0.0123, ...]
]
```

---

### POST `/v1/embeddings`

If request includes a `"model"` field, response is OpenAI-compatible:

Request example:

```json
{
  "model": "mlsa-iai-msu-lab/sci-rus-tiny",
  "input": ["sentence A", "sentence B"]
}
```

Response example:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.0012, ...], "index": 0},
    {"object": "embedding", "embedding": [0.0034, ...], "index": 1}
  ],
  "model": "mlsa-iai-msu-lab/sci-rus-tiny"
}
```

If `"model"` key is absent the endpoint returns the raw list of embeddings (same format as `/`).
