# Parity with the reference implementation

The plugin reads StartLux-Decision in-process through `libllama`. Its answers must be the reference
implementation's (`startlux_decide`, StartLux-Decision v1.1) over a stock `llama-server` of the same
llama.cpp build: the prompt byte for byte, the token ids one for one, the option probabilities within
1e-5.

## Fixtures

- `testdata/cases.json`: Godwit questions about synthetic mail (`scripts/cases.py`): Godwit's seven
  preset label questions on every state, plus choice and score questions, with states sized for each
  context tier (1K, 2K, 4K).
- `testdata/ref/<variant>.json`: the reference's prompt, token ids, letter log-probabilities and option
  probabilities for each case (`scripts/reference.py`).
- `testdata/config/<variant>/decision_config.json`: the model's read-out config.

Real mail never goes in the repository: `scripts/cases.py --states` writes its cases to
`testdata/local/`, which git ignores.

## Making the references

Run the pinned build's `llama-server` (`third_party/llama/VERSION`) with the model's GGUF, as a stock
server reads an Instant model (non-causal context, the whole prompt in one micro-batch):

```bash
llama-server -m model.gguf -c 4096 -b 4096 -ub 4096 --parallel 1 --ctx-checkpoints 0 --cache-ram 0 -ngl 99 --port 8093
```

Then, in an environment with `startlux_decide` and `transformers` installed, with the model's Hugging
Face folder for its tokenizer and `decision_config.json`:

```bash
python scripts/reference.py --tokenizer HF_MODEL_DIR --server http://127.0.0.1:8093 testdata/cases.json testdata/ref/2b-q8.json
```

## Running the tests

`TestParityPrompt` and `TestParityReadout` need only the fixtures. `TestParityEngine` and
`TestLongPrompts` also need the engine and the model:

```bash
STARTLUX_ENGINE=ENGINE_DIR STARTLUX_MODEL=MODEL_DIR STARTLUX_REF=testdata/ref/2b-q8.json go test -run 'Parity|Long' .
```

`ENGINE_DIR` holds the build's libraries as Godwit stores them (`libllama.0.dylib` and its
`libggml*.0.dylib` on macOS); `MODEL_DIR` holds `model.gguf` and `decision_config.json`.
`STARTLUX_CASES` points at another case file (`testdata/local/cases.json`).

## Results

llama.cpp b11707, Apple M5 Max, Metal, 2026-10-17:

| Model | Cases | Prompts | Token ids | Largest probability difference |
| --- | --- | --- | --- | --- |
| 2B Instant Q8_0 | 43 synthetic | identical | identical | 1.22e-6 |
| 2B Instant Q8_0 | 693 (99 real messages × 7 labels, local) | identical | identical | 1.41e-6 |
| 4B Instant Q4_K_M | 43 synthetic | identical | identical | 1.58e-6 |
| 4B Instant Q4_K_M | 693 (local) | identical | identical | 1.36e-6 |

Every prompt is also read again in the largest context it grew to: the context tier changes nothing
beyond 1e-6.
