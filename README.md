# gemma4.c

Gemma 4 E2B CPU inference in 700 lines of pure C.

An educational project made to understand how LLM inference works. The full inference path is implemented in one file without external libraries.

<img width="800" height="339" alt="demo" src="https://github.com/user-attachments/assets/2de47c35-ee34-473e-9164-635c82377267" />

## Benchmark

Measured on an AMD Ryzen 7 7700 using the default native build.

| Implementation | Prefill (pp512) | Decode (tg128) |
| -------------- | --------------: | -------------: |
| gemma4.c (int8) | 638.86 ± 2.93 tok/s | 25.90 ± 0.01 tok/s |
| llama.cpp (Q8_0) | 276.32 ± 1.85 tok/s | 23.83 ± 0.01 tok/s |

Results are the mean ± sample standard deviation over 12 timed runs after one discarded warmup, with each runtime using its fastest tested thread count. Both were built natively for CPU and dynamically quantize matrix inputs to int8.

On an Apple M4 Max, the NEON build measures 637.62 ± 14.31 tok/s prefill and 25.06 ± 0.09 tok/s decode under the same protocol at the default thread count of one per performance core. Decode reaches about 29 tok/s with `OMP_NUM_THREADS=10`, which trades a little prefill throughput. The E4B build measures roughly 333 tok/s prefill and 14.6 tok/s decode on the same machine.

```bash
./run -m ./gemma4-E2B-int8.bin --bench 512 128
```

## Quick start

You need a CPU with AVX2 or ARM NEON dot product instructions (any Apple Silicon Mac), an OpenMP-capable C compiler, and `make`. The model takes about 5.0 GB of disk space, and 8 GB of RAM is recommended.

Clone the repository and download the ready-to-run model:

```bash
git clone https://github.com/ryanssenn/gemma4.c
cd gemma4.c
python3 -m pip install -U huggingface_hub
hf download QmogAI/gemma4-e2b-int8 gemma4-E2B-int8.bin --local-dir .
```

On Linux:

```bash
make
./run -t 0 -n 256 "Why is the sky blue?"
```

On macOS, Apple's compiler needs Homebrew's OpenMP runtime:

```bash
brew install libomp
make
./run -t 0 -n 256 "Why is the sky blue?"
```

On Windows, use a MinGW-w64 environment that provides `gcc`, OpenMP, and `make`:

```powershell
make win64 WINCC=gcc
.\run.exe -t 0 -n 256 "Why is the sky blue?"
```

## Options

- `-m` sets the model path. The default is `gemma4-E2B-int8.bin`.
- `-t` sets the temperature. The default is `1.0`. Use `0` for greedy decoding.
- `-n` sets the maximum number of tokens to generate. The default is `1,024`.
- `--bench` measures prefill and decode throughput.
- `--dump-logits` writes prompt logits as float32 binary data.

## HTTP server

`server.c` wraps the runtime in a minimal OpenAI-style chat completions endpoint: just
enough HTTP, SSE, and JSON for coding harnesses such as [OpenCode](https://opencode.ai),
[Pi](https://github.com/earendil-works/pi), and Prospero, with tool calling prompted
through the chat template and translated to the standard `tools`/`tool_calls` wire
format.

```bash
make server
./server -m ./gemma4-E2B-int8.bin -p 8080
```

It serves streaming `POST /v1/chat/completions`, and `GET /v1/models`, which lists the
one model, `gemma4`, with its context size. Requests are served one at a time, and the KV
cache carries over between requests that share a token prefix, so a growing agent
conversation avoids re-prefilling its history each turn. During a long prefill, or while
a tool call is withheld until complete, the stream sends SSE keep-alive comments every
few seconds, so clients with idle timeouts keep waiting. A client that hangs up mid-prefill
keeps what was prefilled, and its retry resumes from there.

OpenCode (`~/.config/opencode/opencode.json`):

```json
{
  "provider": {
    "gemma4c": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "gemma4.c",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "gemma4": { "name": "Gemma 4 E2B (local)", "limit": { "context": 131072, "output": 4096 } }
      }
    }
  }
}
```

Pi (`~/.pi/agent/models.json`):

```json
{
  "providers": {
    "gemma4c": {
      "name": "gemma4.c",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "unused",
      "models": [
        { "id": "gemma4", "name": "Gemma 4 E2B (local)", "contextWindow": 131072, "maxTokens": 4096 }
      ]
    }
  }
}
```

Prospero (`~/.config/prospero/prospero.ini`) learns the model and its context size from
the listing. Since the server takes one request at a time and holds one KV cache, limit
the endpoint to one request in flight, and since the runtime is text only, keep
screenshots and images out of requests:

```ini
[endpoint gemma4c]
url = http://127.0.0.1:8080/
max_concurrent = 1

[model gemma4]
vision = false
```

Then select `gemma4c/gemma4` in OpenCode's `/models`, run
`pi --provider gemma4c --model gemma4`, or choose `gemma4c/gemma4` in Prospero (or
`prospero run -m gemma4c/gemma4 PROMPT`). Expect small-model results: the protocol round
trip works, but a 2B model is a weak coding agent.

## Model

The C runtime cannot read the original checkpoint directly. `exporter.py` takes the tokenizer and language-model weights from the Hugging Face checkpoint and writes them in the exact layout used by `gemma4.c`.

Matrix weights are stored as int8 with FP16 scales. Inputs to linear layers are dynamically quantized to int8 while the rest of the activations remain float32. The resulting file is about 5.0 GB (4.7 GiB).

The larger Gemma 4 E4B checkpoint is also supported. The compiled-in architecture must
match the model file, so build the E4B runtime and server as `make run-e4b server-e4b`,
then export [google/gemma-4-E4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized)
to `gemma4-E4B-int8.bin` (about 7.4 GiB) with the same exporter command. E4B validates at
96.9% top-1 agreement over 256 positions.

To create it yourself instead:

```bash
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized -o ./gemma4-E2B-int8.bin
```

Python is only needed to export the model or run numerical validation. Once the `.bin` file exists, inference runs entirely through the C program.

## GGUF model

E2B builds also load Google's quantization-aware q4_0 GGUF, a 3.35 GB
download, given its path with `-m`:

```bash
hf download google/gemma-4-E2B-it-qat-q4_0-gguf gemma-4-E2B_q4_0-it.gguf \
  --revision 675cff42a74c774d6cb76f76d8eacb49b48c9b93 --local-dir .
./run -m gemma-4-E2B_q4_0-it.gguf -t 0 "Why is the sky blue?"
```

Only that file is supported, not GGUF in general. Its transformer matrices
are q4_0 and are multiplied in place; its q6_K per-layer embedding table is
read a row at a time; the q6_K token table, which is also the output
projection, and the f16 per-layer projection are converted to int8 at load.
The tokenizer built from its metadata is identical to the exported one.

Its q4_0 values are not exactly the trained ones. The QAT checkpoints'
weights (checked on E4B's) lie exactly on a 4-bit grid, but with each
block's own step, which
the GGUF's quantization (the block's largest value over -8) does not
recover, for a 5% relative error in every matrix. Over the first 1,003
positions of the validation article, it agrees with the int8 model on the
top token 88.4% of the time (mean KL 0.049); with the int8 model's
transformer matrices swapped in, 97.8%.

## Numerical validation

The implementation is validated against the Hugging Face Transformers reference implementation running Google's unquantized [Gemma 4 E2B QAT checkpoint](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) in BF16. Both implementations process the complete Éva Gauthier article from the WikiText-103 validation split under teacher forcing. The passage contains 2,387 text tokens, and the added BOS token brings the comparison to 2,388 model positions.

| Metric | Result |
| ------ | -----: |
| Top-1 agreement | 2,305 / 2,388 (96.5%) |
| Mean KL divergence | 0.005207 |

An exact match is not expected because gemma4.c uses int8 matrix weights and linear inputs while the reference runs the unquantized checkpoint in BF16. The output distributions nevertheless remain closely aligned.

Build the C runtime first (`make` or `make win64`). The complete validation peaks at about 10 GiB of RAM:

```bash
python3 validation.py
```

Pass a smaller token count for a quicker check:

```bash
python3 validation.py 64
```

`validation.py` uses the runtime's `--dump-logits` flag to collect float32 logits after each prompt position. The flag can also be used directly when comparing gemma4.c with another implementation:

```bash
./run -m ./gemma4-E2B-int8.bin --dump-logits "Why is the sky blue?" > logits.bin
```

## Repository contents

- `gemma4.c` contains the tokenizer, model definitions, kernels, transformer, KV cache, and generation loop.
- `server.c` serves the runtime over a minimal OpenAI-style streaming chat completions API with tool calling.
- `exporter.py` converts the original checkpoint into the binary layout read by the C runtime.
- `validation.py` compares the runtime's logits with Hugging Face Transformers.
- `validation.txt` contains the WikiText-103 passage used for numerical validation.
- `win.c` and `win.h` provide the small Windows memory-mapping compatibility layer.
- `Makefile` builds the Linux, macOS, or Windows executable.
