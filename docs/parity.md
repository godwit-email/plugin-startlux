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

### CPU back ends (GitHub-hosted runners)

llama.cpp b11707, 2026-10-19, the 43 synthetic cases against the Metal references above. Runners are
3–4 core virtual machines with no usable GPU (the macOS arm64 runner's Metal is virtualized), so the
times are a floor for slow machines, not a promise for real ones. Peak memory is `maxrss` on macOS and
Linux and peak commit on Windows, at the 4K tier.

| Platform | Model | Largest difference | Flips at 0.8 | p50 1K / 2K / 4K | Peak memory |
| --- | --- | --- | --- | --- | --- |
| macOS arm64 (VM Metal) | 2B Q8 | 2.61e-2 | 0 | 9.0 / 21.5 / 34.7 s | 4.1 GB |
| macOS arm64 (VM Metal) | 4B Q4 | 3.26e-2 | 0 | 16.8 / 43.1 / 89.4 s | 4.4 GB |
| macOS x64 (CPU) | 2B Q8 | 4.47e-3 | 0 | 6.3 / 36.0 / 91.9 s | 2.7 GB |
| macOS x64 (CPU) | 4B Q4 | 3.40e-2 | 0 | 7.4 / 43.2 / 101.2 s | 5.4 GB |
| Linux x64 | 2B Q8 | 1.23e-2 | 0 | 3.2 / 22.3 / 48.3 s | 2.6 GB |
| Linux x64 | 4B Q4 | 3.98e-2 | 0 | 6.8 / 46.9 / 101.8 s | 5.2 GB |
| Linux arm64 | 2B Q8 | 2.33e-2 | 0 (4 options cross 0.5) | 1.2 / 11.0 / 32.0 s | 4.6 GB |
| Linux arm64 | 4B Q4 | 3.63e-2 | 0 | 4.1 / 36.7 / 101.3 s | 6.2 GB |
| Windows x64 | 2B Q8 | 1.17e-2 | 0 | 3.8 / 26.1 / 56.6 s | 8.2 GB |
| Windows x64 | 4B Q4 | 3.06e-2 | 0 | 5.3 / 37.3 / 81.6 s | 10.0 GB |
| Windows arm64 | 2B Q8 | 1.46e-2 | 0 | 0.9 / 6.6 / 15.3 s | 10.2 GB |
| Windows arm64 | 4B Q4 | 4.91e-2 | 0 | 3.3 / 23.6 / 52.8 s | 11.0 GB |

Other back ends are not bit-for-bit the Metal reference: within 1e-5 holds only on the back end the
references were made with. Across back ends the bar is the same decision at the threshold.
