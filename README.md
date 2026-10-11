# agent-harness

A minimal, model-agnostic coding-agent harness in Python (Anthropic, OpenAI, Groq),
built to study which scaffolding choices actually matter.

Status: work in progress (week 1).

## Setup
    python -m venv .venv
    pip install -e ".[dev]"
    cp .env.example .env   # then add your keys