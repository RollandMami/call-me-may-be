*This project has been created as part of the 42 curriculum by mamiandr.*

# Call Me Maybe — Function Calling with Constrained Decoding

## Description

**Call Me Maybe** is an introduction to *function calling* in Large Language Models (LLMs). Instead of answering a request in free text, the program translates a natural-language prompt into a **structured, machine-executable function call**.

For example, given the prompt:

```
What is the sum of 265 and 345?
```

the program does **not** answer `610`. It produces:

```json
{
  "prompt": "What is the sum of 265 and 345?",
  "name": "fn_add_numbers",
  "parameters": { "a": 265, "b": 345 }
}
```

### Goal

Small models such as **Qwen3-0.6B** (the model used here, ~0.6 billion parameters) are unreliable when simply asked to "output valid JSON": they add commentary, break the syntax, invent function names or return values of the wrong type. The goal of this project is to reach a **100 % structurally valid output** with such a small model, by **constraining the decoding process itself** rather than hoping the model behaves.

### Overview

The pipeline, for each prompt of the test file, is:

1. **Load & validate** the function catalog (`functions_definition.json`) with Pydantic.
2. **Select the function**: the model generates the function name token by token, but at each step the logits are masked so that only tokens that can still lead to a name from the catalog are allowed.
3. **Extract the arguments**: the JSON skeleton (`"key": `) is *forced* by the program, and the model is only allowed to generate the *value*, under a type-specific constraint (numeric grammar for `number`/`integer`, single-line string terminated by an unescaped quote for `string`).
4. **Write** all results to a JSON file (`data/output/function_calls.json`).

The project uses only `numpy`, `pydantic`, `python-dotenv` and a provided wrapper (`llm_sdk`) around Hugging Face `transformers`. No constrained-decoding library (outlines, guidance, etc.) is used: the logit masking is implemented from scratch.

### Features

- Constrained function-name selection (the output is *always* a function that exists in the catalog, or the prompt is reported as unmatched).
- Type-aware argument generation (`number`, `integer`, `string`).
- Strict validation of the function definitions (valid identifiers, unique names, supported types, no extra fields).
- Clear error messages for missing files, malformed JSON or invalid schemas.
- Debug mode that prints the step-by-step decoding trace for the first prompt.
- Fully typed code, checked with `mypy` and `flake8`.

---

## Instructions

### Requirements

- Python **>= 3.10**
- [`uv`](https://docs.astral.sh/uv/) (dependency and virtual-environment manager)
- Internet access on the first run (to download the model from the Hugging Face Hub)
- Enough disk space for the model and its dependencies (`torch`, `transformers`). The `Makefile` stores everything in `~/sgoinfre/$USER` (cache, Python, virtualenv) to avoid filling the home directory on 42 computers.

### Installation

```bash
git clone <repository_url> call-me-may-be
cd call-me-may-be

# (optional) provide a Hugging Face token to avoid anonymous rate limits
cp .env.exemple .env
# then edit .env and set:  HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"

make install
```

`make install` creates the cache directories and runs `uv sync`, which installs the dependencies declared in `pyproject.toml` (including the local `llm_sdk` package, installed in editable mode).

### Execution

```bash
make run
```

This is equivalent to `uv run -q python -m src` and uses the default paths below. You can override any of them:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input                data/input/function_calling_tests.json \
    --output               data/output/function_calls.json
```

| Option                   | Default                                    | Description                                   |
| ------------------------ | ------------------------------------------ | --------------------------------------------- |
| `--functions_definition` | `data/input/functions_definition.json`     | Catalog of the callable functions             |
| `--input`                | `data/input/function_calling_tests.json`   | List of prompts to translate                  |
| `--output`               | `data/output/function_calls.json`          | Where the results are written (dirs created)  |
| `--debug`                | off                                        | Process only the first prompt and trace decoding |

### Makefile targets

| Target             | Action                                                                  |
| ------------------ | ----------------------------------------------------------------------- |
| `make install`     | Create caches and install dependencies with `uv sync`                   |
| `make run`         | Install (if needed) and run the program                                 |
| `make debug`       | Run under `pdb` with `--debug` (first prompt only, decoding trace)      |
| `make lint`        | Run `mypy` (with the required flags) and `flake8`                       |
| `make lint-strict` | Run `mypy --strict` and `flake8`                                        |
| `make clean`       | Remove `__pycache__`, `.mypy_cache`, `.pytest_cache` and `data/output`  |
| `make fclean`      | `clean` + remove the virtual environment                                |

### Project structure

```
call-me-may-be/
├── data/
│   └── input/
│       ├── functions_definition.json   # function catalog
│       └── function_calling_tests.json # prompts to process
├── llm_sdk/                            # provided wrapper around Hugging Face (Qwen3-0.6B)
├── src/
│   ├── __main__.py     # entry point / orchestration
│   ├── parser.py       # CLI arguments and input-file checks
│   ├── validation.py   # Pydantic models for the function catalog
│   ├── prompts.py      # ChatML prompt builders (selection + extraction)
│   ├── tokens.py       # vocabulary helpers (candidate / numeric / stop tokens)
│   └── answer.py       # constrained decoding (function name + arguments)
├── Makefile
├── pyproject.toml
├── mypy.ini / .flake8
├── .env.exemple        # template for the Hugging Face token
└── README.md
```

---

## Example usage

### Input files

`data/input/functions_definition.json` (excerpt):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": { "a": { "type": "number" }, "b": { "type": "number" } },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

`data/input/function_calling_tests.json` (excerpt):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Replace all vowels in 'Programming is fun' with asterisks" }
]
```

### Running

```console
$ make run
Function Calling System: {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
Function Calling System: {"name": "fn_greet", "parameters": {"name": "shrek"}}
...
```

### Output file

`data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

### Debug mode

```console
$ uv run python -m src --debug
>>> 'fn' -> fn
>>> '_add' -> fn_add
>>> '_numbers' -> fn_add_numbers
Function Calling System: {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

Each line shows the token chosen under the mask and the accumulated prefix of the function name. (The exact token split depends on the Qwen tokenizer; the trace above is illustrative.)

### Error handling

```console
$ uv run python -m src --input missing.json
Input file not found

$ uv run python -m src --functions_definition bad.json
Invalid definitions in bad.json:
  [0 -> name] Value error, Invalid function name: 1abc
```

---

## Algorithm explanation

### Why constrained decoding?

At every step, a causal LLM outputs one **logit per token of its vocabulary** (about 151 000 for Qwen3). Normally the next token is sampled or chosen with `argmax`. With **constrained decoding**, before choosing, we compute the set of tokens that are *legal* at this position and set the logits of all the others to `-inf`. The model is then free to choose *among valid continuations only*, so the output cannot violate the structure — whatever the model "wants" to say.

```
logits  ──►  + mask (0 for allowed tokens, -inf for the others)  ──►  argmax  ──►  next token
```

### Step 1 — Function selection (`get_func`)

1. A ChatML prompt lists every function as `- name(param: type, ...): description` and asks the model to reply with the function name only (or `Null`).
2. `build_candidate_tokens` pre-computes, once at start-up, every vocabulary token whose text is a **substring of at least one function name**. Only these tokens can ever be used while generating a name.
3. At each step, with `current_prefix` being what was generated so far, a token `t` is allowed if **some function name starts with `current_prefix + t`**.
4. If `current_prefix` is already a complete function name, the end-of-sequence token (`<|im_end|>`) becomes legal too (needed when a name is the prefix of another one).
5. The logits are masked, `argmax` picks the token, and it is appended to both the prefix and the model input.
6. Decoding **stops early** as soon as the prefix equals a function name that is not a strict prefix of another one (the choice is then unambiguous), or when the model emits EOS.
7. If the final string is not in the catalog, `None` is returned and the prompt is reported as unmatched instead of producing an invalid call.

Because the allowed set is derived from the actual catalog, the result is **guaranteed to be an existing function** — it works with any catalog, not only the five provided functions.

### Step 2 — Argument extraction (`get_arguments`, `generate_value`)

The selected function schema is injected into a second prompt (`build_args_prompt`) containing two few-shot examples. The assistant turn is **pre-filled** with:

```
{"function": "fn_xxx", "arguments": {
```

Then, for each parameter *in declaration order*:

1. The program **forces** the structural text — `"key": ` (preceded by `, ` after the first parameter), plus an opening `"` for strings — by encoding it and appending it directly to the input. The model never has to generate keys, colons, commas or braces, so it cannot get them wrong.
2. The model generates **only the value**, under a constraint that depends on the declared type:

| Type                | Allowed tokens at each step                                                                                                        | Termination                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `number`            | Numeric tokens (`[0-9.\-]+`) such that `text + token` still matches the prefix regex `-?(\d+(\.\d*)?)?`                           | Once `text` fully matches `-?\d+(\.\d+)?`, the stop tokens `,` / `}` or EOS become legal; picking one ends the value |
| `integer`           | Numeric tokens such that `text + token` matches `-?\d*`                                                                            | Same as above with `-?\d+`                                                |
| `string`            | Any token **not containing a newline**, plus EOS                                                                                   | The first **unescaped** `"` in the generated text closes the string        |

3. The raw text is converted to its declared Python type: `int(...)`, `float(...)` or — for strings — unescaped via `json.loads` (so that `\\d+` or `\"` inside a value are decoded correctly, with a fallback to the raw text).
4. The closing `"` is re-injected into the model input before the next parameter.

### Handling strings that end in the middle of a token

BPE tokens can straddle the closing quote (for example a token `"}` or `."`). When the closing quote is found *inside* the last generated token, the program keeps only the part of the text **before the quote**, re-encodes the kept tail of that token, and appends it to the input, so that the context seen by the model by the next parameter is consistent with the value that was actually retained.

### Complexity

For a prompt that selects a function with `k` parameters, the model is called once per generated token (function name tokens + value tokens). `MAX_TOKENS = 100` bounds the number of steps for the name and for each value, so decoding always terminates.

---

## Design decisions

- **Two-stage generation (function, then arguments).** Selecting the name first lets the second prompt contain *only* the relevant schema, which is shorter and less ambiguous for a 0.6B model, and allows the argument constraints to be built from the exact types of the chosen function.
- **Forcing the JSON skeleton instead of generating it.** Everything that is not a value (keys, quotes, colons, commas) is injected by the program. This removes entire classes of errors and saves model calls.
- **Prefix-matching over the catalog rather than a generic JSON grammar.** A small, dedicated finite-state check (prefix of a known name; regex prefix for numbers; quote detection for strings) is simple, fast to evaluate on each token and easy to reason about.
- **Candidate-token pre-computation.** The vocabulary is filtered once, at start-up, to the tokens that can appear in a function name, which keeps the per-step cost of `get_func` low.
- **Pre-filled assistant turn and few-shot examples.** The assistant answer begins with the JSON prefix and two examples demonstrate the expected format, including a regex example (`\\d+`) that shows how backslashes must be escaped.
- **Greedy decoding (`argmax`).** The output is deterministic: the same input always gives the same result, which makes testing and debugging easier.
- **Pydantic validation of the catalog.** Names and parameter names must be valid identifiers, function names must be unique, types are restricted to `number`, `integer`, `string`, `boolean`, extra fields are rejected and models are frozen. Errors are rewritten into readable messages (`[path] message`).
- **No-match handling.** If no allowed function can be produced, the prompt is skipped with a message instead of emitting a wrong call.
- **Strict typing and linting.** All functions are annotated and documented (Google-style docstrings) and the code passes `mypy` and `flake8` (79 columns).
- **Everything stored in `sgoinfre`.** The `Makefile` redirects the Hugging Face, `uv` and virtualenv directories there, which is necessary on the school machines where the home quota is small.

---

## Performance analysis

> The qualitative points below come from the design of the algorithm; the measured values (accuracy and timings) depend on the machine and on the test set, and are given as a template to complete with your own runs.

### Reliability

- **Structural validity: by construction.** The function name always belongs to the catalog, keys are written by the program in the right order, numeric values always match a numeric grammar and strings are always properly closed. The output JSON is therefore well-formed for every processed prompt.
- **Deterministic.** Greedy decoding gives identical results from run to run.
- **Robust to malformed inputs.** Missing files, invalid JSON, wrong types or duplicated names are detected before any model call, with an explicit error.

### Accuracy

Constrained decoding guarantees the **form** of the output, not its **semantic correctness**: the model may still choose the wrong function or extract a wrong value. In practice, the 0.6B model performs well on the provided cases because the function descriptions are clear, the schema is injected in the prompt and few-shot examples are provided. The most delicate cases are the **regex-based substitutions** (e.g. "Replace all vowels … with asterisks"), where the model must invent a correct regular expression such as `[aeiouAEIOU]`.

| Metric                                       | Result            |
| -------------------------------------------- | ----------------- |
| Valid JSON / schema compliance               | 100 % (by design) |
| Correct function selected                    | _to fill in_ / 11 |
| Fully correct arguments                      | _to fill in_ / 11 |

### Speed

- The model is run **once per generated token**, with the full sequence given as input (no KV-cache is exposed by the provided SDK). The cost grows with the prompt length and with the number of generated tokens.
- On a CPU, the model loading and the first run (download included) dominate; subsequent runs reuse the Hugging Face cache stored in `sgoinfre`.
- Early stopping (unique function name, stop token after a number) and the forced skeleton reduce the number of model calls.
- Known overheads: for strings, the allowed-token set is rebuilt at each step by scanning the full decoded vocabulary; for functions, the candidate set is filtered at each step. The vocabulary decoding (`string_texts`) is done once at start-up.

| Metric                                       | Result            |
| -------------------------------------------- | ----------------- |
| Model loading time                           | _to fill in_      |
| Average time per prompt                      | _to fill in_      |
| Total time for the 11 provided prompts       | _to fill in_      |

### Known limitations

- The `boolean` type is accepted by the schema validator, but the generation step handles only `number`, `integer` and `string`.
- Strings are single-line (newlines are forbidden in generated values).
- A prompt that matches no function is skipped (reported on the console, absent from the output file).
- Functions with the same prefix are handled, but very long names or very large catalogs increase the per-step cost.

---

## Challenges faced

| Challenge                                                          | Solution                                                                                                                                       |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| The small model produced extra text, wrong names or broken JSON    | Moved from "prompt and parse" to **logit masking**, then to **forcing the JSON skeleton** so that only values are generated                    |
| Function names that are a prefix of another one                    | Allow EOS only when the current prefix is a full name, and stop early only if no longer name starts with the prefix                            |
| Numbers: tokens such as `-`, `.`, `12.`, `007`                     | Prefix regex checked at each step on `text + token`, and a separate full-match regex to know when the value may legally end                    |
| Knowing when a number ends                                         | Stop tokens `,` and `}` (and EOS) are only authorised once the text is a complete number                                                       |
| Strings ending inside a multi-character token                      | Detect the first unescaped quote in the accumulated text, cut the value there and re-encode only the kept part of the last token               |
| Escape sequences in regex arguments (`\d+`)                        | Decode the raw string with `json.loads`, and show the escaped form in the few-shot example                                                     |
| Tokens with leading spaces / byte-level BPE                        | Candidate and value texts come from `model.decode` / the vocabulary, and the matching is done on decoded text instead of raw token identifiers |
| Disk quota and long downloads on school computers                  | `Makefile` redirects `HF_HOME`, `UV_CACHE_DIR`, the Python install and the virtualenv to `sgoinfre`, with larger `uv` HTTP timeout and retries |
| Reporting configuration errors clearly                             | Pydantic validators + a loader that converts `ValidationError` and I/O errors into readable messages                                          |

---

## Testing strategy

The program is validated at several levels:

1. **Static checks.** `make lint` (mypy with `--warn-return-any --warn-unused-ignores --disallow-untyped-defs --check-untyped-defs` + flake8) and `make lint-strict` (`mypy --strict`) must pass without error.
2. **Schema validation tests.** Invalid catalogs were fed to the loader to check the error messages: missing file, malformed JSON, root that is not a list, invalid function / parameter identifier, unsupported type, extra field, empty name or description, duplicated function names.
3. **End-to-end run on the provided test set.** `make run` on the 11 provided prompts, covering the 5 functions and the 3 parameter kinds: addition of integers (`2 + 3`, `265 + 345`), greetings, string reversal, square root (`16`, `144`) and three regex substitutions (digits, vowels, a word).
4. **Manual inspection of the decoding.** `make debug` (or `--debug`) displays, token by token, the prefix being built while the function name is selected, which allows checking that the mask lets the right candidates through.
5. **Edge cases tried.**
   - Prompts that match no function (the prompt is skipped and reported).
   - Negative and decimal numbers.
   - Strings containing quotes, backslashes, digits and punctuation.
   - Catalogs with function names that share a prefix.
   - Missing input files and bad CLI paths.
   - An output directory that does not exist yet (created automatically).
6. **Determinism check.** Running the program twice on the same inputs gives identical output files (greedy decoding).

To add your own test case, append `{"prompt": "..."}` to `data/input/function_calling_tests.json` (and, if needed, a new function to `functions_definition.json`), then run `make run` and inspect `data/output/function_calls.json`.

---

## Resources

### Documentation and references

- [Qwen3 model card (`Qwen/Qwen3-0.6B`)](https://huggingface.co/Qwen/Qwen3-0.6B) — the model used by the project.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers) — tokenizers, causal LMs, logits and generation.
- [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub) — model download and caching (`hf_hub_download`).
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — models, validators and `ConfigDict`.
- [NumPy documentation](https://numpy.org/doc/) — array operations used for the logit masks.
- [uv documentation](https://docs.astral.sh/uv/) — dependency management and virtual environments.
- [mypy documentation](https://mypy.readthedocs.io/) and [flake8 documentation](https://flake8.pycqa.org/) — static typing and style checks.
- [Python `argparse`](https://docs.python.org/3/library/argparse.html), [`re`](https://docs.python.org/3/library/re.html) and [`json`](https://docs.python.org/3/library/json.html) modules.

### Articles and background reading

- Vaswani et al., *Attention Is All You Need* (2017) — the Transformer architecture.
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016) — Byte-Pair Encoding.
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023) — constrained generation with finite-state machines (the idea behind the `outlines` library).
- Hugging Face blog, *How to generate text: using different decoding methods for language generation* — greedy search, beam search and sampling.
- OpenAI / Anthropic documentation on *function calling* / *tool use* — the use case this project reproduces at a small scale.
- [JSON Schema](https://json-schema.org/) — the type vocabulary (`number`, `integer`, `string`, `boolean`) used in the function definitions.

### Use of AI

AI assistants were used as a **support tool**, not as a substitute for understanding the subject. All the code was reviewed, tested and understood before being kept.

| Task                                                                                       | How AI was used                                                                                     |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Understanding the concepts (logits, tokenization, BPE, constrained decoding)               | Explanations and discussion of approaches                                                           |
| Reviewing the algorithm and edge cases (strings ending inside a token, number termination) | Suggestions of cases to test and of possible failure modes                                          |
| Docstrings and code documentation                                                          | Help to draft and harmonise the Google-style docstrings                                             |
| Linting / typing issues                                                                    | Help to interpret `mypy` and `flake8` messages                                                      |
| Writing this `README.md`                                                                   | Drafting and structuring of the document from the project source code, then reviewed and corrected  |

The core of the project — the decoding strategy, the masks and the implementation choices — was designed and written by the author.
