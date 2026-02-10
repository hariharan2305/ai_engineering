# Meta Prompting: DSPy Framework

This directory contains learning materials and experiments with **DSPy** — Stanford NLP's framework for programming (not prompting) language models.

## What is DSPy?

DSPy shifts prompt engineering from manual string manipulation to structured, optimizable Python code. Instead of crafting prompts by hand, you define *what* the model should do (Signatures), *how* it should think (Modules), and let optimizers find the best prompts automatically.

## Contents

| File | Description |
|------|-------------|
| [DSPy_Complete_Guide.md](./DSPy_Complete_Guide.md) | Comprehensive reference covering all DSPy concepts, modules, optimizers, and best practices |
| [DsPy_basics.md](./DsPy_basics.md) | Initial research notes and decision frameworks for when to use DSPy |
| [DSPy_Intro.ipynb](./DSPy_Intro.ipynb) | Jupyter notebook with hands-on code examples |
| [main.py](./main.py) | Python script examples |

## Key Takeaway

> DSPy is for **building products**, not one-off queries. You need training data, a metric, and scale to benefit from it.

## Setup

```bash
# Create virtual environment
uv venv

# Install dependencies
uv sync

# Or with pip
pip install dspy
```

## Resources

- [DSPy Documentation](https://dspy.ai)
- [GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Stanford NLP](https://nlp.stanford.edu/)
