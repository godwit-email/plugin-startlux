# StartLux for Godwit

A native [Godwit](https://github.com/godwit-email/godwit) plugin that answers Godwit's judge questions
(mail labels, yes/no and choice questions) with [StartLux-Decision](https://github.com/StartLuxLabs/Startlux-Decision)
models on this computer. Nothing is sent anywhere.

It offers two models, chosen in Godwit's judge model setting and downloaded only when picked:

| Model | Download | Memory | Agreement with gpt-6-luna on 99 messages |
| --- | --- | --- | --- |
| StartLux Decision 4B Q4 (`decision-4b-q4`) | 2.7 GB | 8 GB | 92.1% at 0.5, 90.2% at 0.8 |
| StartLux Decision 2B Q8 (`decision-2b-q8`) | 2.0 GB | 6 GB (Windows: still to be measured) | 88.9% at 0.5, 85.4% at 0.8 |

The numbers are agreement with a reference model on one inbox (godwit-email/godwit#64), not general
accuracy. StartLux reads best at thresholds of 0.5 to 0.7; Godwit's default is 0.8.

## How it works

Godwit downloads the pinned llama.cpp libraries and the chosen model as registry assets, checks every
file against the signed registry, and hands the plugin the verified directories in `models.load`. The
plugin loads `libllama` from there (`engine.c`), renders each question in the format the models were
trained on (`prompt.go`), reads the whole prompt in one pass and turns the option letters' logits into
answers (`readout.go`). One prompt is read at a time; the context is the smallest of 1K, 2K and 4K
tokens that holds the prompt.

Methods (ABI v1, see `sdk/godwit_plugin.h` and Godwit's plugin model API design):
`describe`, `configure` (no settings), `models.list`, `models.load {modelId, assets}`,
`models.judge {modelId, input, questions}`.

## Building

```bash
go build -buildmode=c-shared -o tmp/plugin.dylib .
python3 scripts/smoke.py tmp/plugin.dylib --godwit GODWIT_CHECKOUT --engine ENGINE_DIR --model MODEL_DIR
```

The plugin links nothing from llama.cpp; it vendors the headers of the pinned build in
`third_party/llama` (MIT). `docs/parity.md` shows how its answers are checked against the reference
implementation.

## Licenses

The plugin is Apache-2.0. The StartLux-Decision weights are CC BY-NC 4.0; llama.cpp is MIT.
