# llm-data-intake

Recruitment Intake Assistant asks candidates about experience, availability,
work authorization, and salary expectations. Candidates can answer in text or
voice, volunteer details early, and correct the final summary.

A model extracts facts from each reply, a verifier checks them, and Python code
decides what to ask next. Each session produces a structured record and a readable
recruiter summary with supporting quotes.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Configure **one provider**. This selects where model requests are sent.

```bash
# OpenAI: use --provider openai
export OPENAI_API_KEY="your-api-key"

# Or Azure OpenAI: use --provider azure
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment"
```

To use Jev as the verifier, also set:

```bash
export TYPESAFE_API_KEY="your-api-key"
```

The examples below use OpenAI. Replace `--provider openai` with
`--provider azure` to use Azure OpenAI. The CLI still requires an explicit provider.

## Run the app

```bash
# Text conversation in the terminal
python -m src --provider openai

# Browser conversation: open the URL printed in the terminal
python -m src --provider openai --mode web

# Use Jev to verify facts extracted by the OpenAI model
python -m src --provider openai --verifier jev
```

By default, the selected provider model handles both extraction and verification.
Jev replaces only verification, never extraction.

The default role is `software-engineer`. Use `--role data-engineer` or another
ID from `data/input/roles.json`. For voice, use `--mode voice`; it requires a
microphone, speakers, and `OPENAI_API_KEY`, even when extraction uses Azure.
Browser mode runs locally on port 8000 (`--port` changes it); keep the terminal open.

Each run prints a session ID. Progress is saved after every reply:

```bash
python -m src --provider openai --resume SESSION_ID
```

Each session saves two files under `data/output/<session-id>/`: `session.json`
contains the facts (`record`), messages (`history`), and resumable state;
`summary.md` is the formatted recruiter handoff with candidate quotes. Use `--output` to change the base directory.
Separate fact and conversation exports are no longer written automatically.

## Run local tests

These tests use simulated model responses and do not make paid API calls.

```bash
python -m unittest discover -s tests
python -m ruff check .
```

## Run live evaluations

These make paid API calls. There are **two separate evaluation commands**:

| Command | What it tests |
| --- | --- |
| `evals.compare_verifiers` | Can each verifier correctly accept or reject a prepared fact? |
| `evals.conversations` | Can the whole app complete a scripted intake with the correct final record? |

### Compare verification decisions

`compare_verifiers` always tests **both the provider model and Jev** on the same
prepared facts. It does not run extraction or conversations. Both provider
credentials and `TYPESAFE_API_KEY` are required.

Choose which set of examples to test with `--suite`:

| Suite | Examples |
| --- | --- |
| `benchmark` (default) | 120 development cases covering clear, ambiguous, and misleading statements |
| `holdout` | 50 separate fixed cases, now reused across runs, for checking whether improvements generalize |
| `corrections` | 7 cases covering genuine correction requests and requests that should be rejected |

```bash
python -m evals.compare_verifiers --provider openai --suite benchmark
python -m evals.compare_verifiers --provider openai --suite holdout
python -m evals.compare_verifiers --provider openai --suite corrections
```

Results show correct decisions, false accepts, false rejects, and consistency.
High-risk cases are repeated twice by default. Add `--json` for detailed results.
Completing this command does not mean every decision was correct; inspect its scores.

### Test complete conversations

`conversations` plays a scripted candidate in three scenarios: answering questions
one at a time, volunteering details early, and correcting salary during review.
It runs real extraction and verification, then checks the final record, completion,
and number of turns. The scripted candidate refuses to confirm an incorrect record.

Here, `--verifier` accepts one name or a comma-separated list, run in the order given:

| Option | Extraction | Verification |
| --- | --- | --- |
| `provider` (default) | Selected provider model | Same provider model |
| `jev` | Selected provider model | Jev |
| `provider,jev` | Selected provider model | Runs each scenario with the provider verifier, then with Jev |

```bash
# Test the default configuration
python -m evals.conversations --provider openai

# Test the configuration that uses Jev as verifier
python -m evals.conversations --provider openai --verifier jev

# Compare the selected verifiers
python -m evals.conversations --provider openai --verifier provider,jev
```

When selecting multiple verifiers, identical inputs reuse the same extraction,
and each turn records their decisions on those exact facts. Each verifier controls
its own conversation, so later questions can differ. Any list containing `jev`
requires `TYPESAFE_API_KEY`. The app itself uses one verifier per session;
comma-separated lists are for conversation evaluations.

The terminal table shows pass/fail, record correctness, turns, repeated answers,
and elapsed time for each scenario and verifier. Add `--json` for detailed output.
The command exits with code 1 if a scenario fails. Reports, messages, extracted
facts, and verification decisions are saved under
`data/output/evaluations/<run-id>/`. Use `--output` to change the base directory.

These are scripted checks, not a substitute for testing with people. See
[recorded evaluation results](evals/README.md) for the latest measured results.
