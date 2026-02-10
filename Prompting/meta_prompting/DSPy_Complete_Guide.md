# DSPy: The Complete Guide to Programming Language Models

> **DSPy** = **D**eclarative **S**elf-improving **Py**thon

DSPy is a framework developed at Stanford NLP that transforms how we build AI systems. Instead of manually crafting and tweaking prompt strings, DSPy lets you write **structured Python code** that the framework then "compiles" into optimized prompts.

**Repository:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
**Documentation:** [dspy.ai](https://dspy.ai)
**License:** MIT

---

## Table of Contents

1. [Philosophy & Core Concept](#1-philosophy--core-concept)
2. [Installation](#2-installation)
3. [The Three-Stage Development Framework](#3-the-three-stage-development-framework)
4. [Core Building Blocks (The "Layers")](#4-core-building-blocks-the-layers)
   - [Signatures](#a-signatures-dspysignature)
   - [Modules](#b-modules-dspymodule)
5. [Optimizers (Formerly "Teleprompters")](#5-optimizers-formerly-teleprompters)
   - [Few-Shot Optimization](#a-for-few-shot-optimization-selecting-examples)
   - [Instruction Optimization](#b-for-instruction-optimization-rewriting-the-prompt)
   - [Advanced Optimizers](#c-advanced-optimizers)
6. [Metrics (The "Loss Function")](#6-metrics-the-loss-function)
7. [When to Use DSPy](#7-when-to-use-dspy)
8. [Decision Matrix: Choosing the Right Optimizer](#8-decision-matrix-choosing-the-right-optimizer)
9. [Practical Examples](#9-practical-examples)
10. [Best Practices & Tips](#10-best-practices--tips)

---

## 1. Philosophy & Core Concept

### The Problem with Traditional Prompting

Traditional prompt engineering involves:
- Manually tweaking prompt strings
- Hoping prompts generalize across different inputs
- Brittle systems that break with model updates
- No systematic way to improve

### The DSPy Approach

DSPy treats **prompt engineering like machine learning**:

| ML Concept | DSPy Equivalent |
|------------|-----------------|
| Model Architecture | Signatures + Modules |
| Forward Pass | Module execution |
| Loss Function | Metrics |
| Training | Optimization |
| Learned Weights | Optimized prompts + demos |

**Key Insight:** Rather than specifying *how* to prompt LLMs, you declare *what* the model needs to accomplish. DSPy figures out the optimal prompting strategy.

---

## 2. Installation

```bash
# Standard installation
pip install dspy

# Latest development version
pip install git+https://github.com/stanfordnlp/dspy.git
```

### Basic Setup

```python
import dspy

# Configure your LLM
lm = dspy.LM('openai/gpt-4o-mini', api_key='your-api-key')
dspy.configure(lm=lm)

# Enable usage tracking (optional)
dspy.configure(track_usage=True)
```

---

## 3. The Three-Stage Development Framework

DSPy structures AI system building into three sequential phases:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. PROGRAMMING │ ──► │  2. EVALUATION  │ ──► │ 3. OPTIMIZATION │
│                 │     │                 │     │                 │
│ Define task,    │     │ Establish       │     │ Use optimizers  │
│ constraints,    │     │ metrics and     │     │ to refine       │
│ initial design  │     │ dev datasets    │     │ prompts/weights │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 4. Core Building Blocks (The "Layers")

These are the static components you write to define *what* the system does.

### A. Signatures (`dspy.Signature`)

A signature is a **declarative specification of input/output behavior**. It abstracts away the prompt text.

**Purpose:** Defines input/output fields and their types. Field names carry semantic meaning—a `question` differs fundamentally from an `answer`.

#### Inline Signatures (Simple & Concise)

```python
# Basic form
"question -> answer"

# With type hints
"context: list[str], question: str -> answer: str"

# Multiple outputs
"question, choices: list[str] -> reasoning: str, selection: int"
```

#### Class-based Signatures (For Complex Descriptions/Constraints)

```python
class EmotionDetector(dspy.Signature):
    """Classify the emotion of the text."""
    sentence: str = dspy.InputField()
    label: str = dspy.OutputField(desc="One of: Joy, Sadness, Anger")

class QAWithContext(dspy.Signature):
    """Answer questions based on provided context."""
    context: list[str] = dspy.InputField(desc="Retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise factual answer")
```

#### Field Configuration

| Field Type | Purpose | Parameters |
|------------|---------|------------|
| `dspy.InputField()` | Define inputs | `desc` (optional hint about input nature) |
| `dspy.OutputField()` | Define outputs | `desc` (optional constraints/format) |

**Supported Types:** `str` (default), `list`, `dict`, `bool`, `Literal`, custom types, `dspy.Image`

---

### B. Modules (`dspy.Module`)

Modules are the "layers" that process Signatures. They determine *how* the model thinks.

#### Core Modules

| Module | Description | Use Case |
|--------|-------------|----------|
| `dspy.Predict` | Basic predictor. No reasoning, just input → output | Simple transformations, fast inference |
| `dspy.ChainOfThought` | Generates "Rationale" before final answer | **Use for 90% of tasks** |
| `dspy.ReAct` | Agentic loop: Thought → Action → Observation → Thought | Tool use, API calls, calculators |
| `dspy.ProgramOfThought` | Outputs executable code whose results inform the response | Math, data processing |
| `dspy.MultiChainComparison` | Runs multiple times, compares outputs to pick best | Factual accuracy, critical decisions |
| `dspy.RLM` | Recursive LM for large contexts via sandboxed execution | Long document processing |

#### Module Usage Pattern

```python
# 1. Declare the module with a signature
classify = dspy.ChainOfThought("text -> category")

# 2. Call it with input arguments
result = classify(text="DSPy is a framework for LLM programming")

# 3. Access output fields
print(result.category)
print(result.reasoning)  # ChainOfThought adds this automatically
```

#### Composing Programs (Building Complex Systems)

```python
class MultiHopQA(dspy.Module):
    def __init__(self):
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # First hop: generate search query
        query = self.generate_query(question=question)

        # Retrieve context (your retrieval logic here)
        context = retrieve(query.search_query)

        # Second hop: generate answer
        return self.generate_answer(context=context, question=question)
```

#### Utility Functions

- `dspy.majority` — Performs voting across multiple predictions

---

## 5. Optimizers (Formerly "Teleprompters")

This is the "Meta" part. Optimizers take your modules and "train" them by rewriting prompts and selecting examples.

**All optimizers accept three inputs:**
1. A DSPy program
2. A metric function
3. Training data (can start with just 5-10 examples)

### A. For "Few-Shot" Optimization (Selecting Examples)

These optimizers focus on finding the perfect "Examples" to put in the prompt context window.

#### `dspy.LabeledFewShot`
- **How it works:** Constructs few-shot examples directly from labeled data
- **Parameters:** `k` (number of examples), `trainset`
- **Usefulness:** When you have high-quality labeled examples ready to use

#### `dspy.BootstrapFewShot`
- **How it works:** Acts like a "Teacher." Runs your `trainset` through the model. If the model gets an answer *right* (according to your metric), it saves that "Question + Reasoning + Answer" combo as a demonstration.
- **Parameters:** `max_bootstrapped_demos`, `max_labeled_demos`
- **Usefulness:** The **Standard starter**. Good for almost everything with 10-50 examples.

#### `dspy.BootstrapFewShotWithRandomSearch`
- **How it works:** Similar to above, but randomly shuffles and tries different combinations of examples to find which *set* gives the highest score.
- **Parameters:** `num_candidate_programs`, `num_threads`
- **Usefulness:** Use when you have more data (50+ examples) and want to squeeze out 2-3% more accuracy.

#### `dspy.KNNFewShot`
- **How it works:** Uses k-nearest neighbors to identify the most relevant training demonstrations for each input.
- **Usefulness:** When examples vary significantly and relevance matters.

---

### B. For "Instruction" Optimization (Rewriting the Prompt)

These are the heavy hitters for creative/textual use cases. They rewrite the actual system instructions.

#### `dspy.COPRO` (COordinate PRompt Optimization)
- **How it works:** Uses an "Optimizer LLM" to propose textual improvements to your prompt prefix. Generates and refines new instructions for each step using coordinate ascent.
- **Usefulness:** Good for zero-shot or very few examples. Great for **style transfer** and **tone adjustment** (e.g., "Make it sound like a Wall Street analyst").

#### `dspy.MIPROv2` (Multi-Instruction Proposal Optimizer)
- **How it works:** Bayesian optimizer that proposes *multiple different instruction styles* AND selects different few-shot examples simultaneously. Runs a massive search to find the best Instruction + Example combo.
- **Usefulness:** **State of the Art** for DSPy. Use with 50+ training examples. Data-hungry but produces the best prompts.

#### `dspy.SIMBA`
- **How it works:** Uses stochastic sampling to identify challenging examples and generate improvement rules.
- **Usefulness:** When you want to focus on edge cases and failure modes.

#### `dspy.GEPA`
- **How it works:** Leverages LM reflection on program trajectories to propose better prompts.
- **Usefulness:** For iterative refinement based on execution traces.

---

### C. Advanced Optimizers

#### `dspy.BootstrapFinetune`
- **How it works:** Distills a prompt-based DSPy program into actual model weight updates.
- **Usefulness:** Production efficiency—move from prompting to fine-tuning.

#### `dspy.Ensemble`
- **How it works:** Combines multiple DSPy programs into a single unified program.
- **Usefulness:** When you have multiple working approaches and want to combine them.

---

### Optimizer Selection Quick Reference

| Your Situation | Recommended Optimizer |
|----------------|----------------------|
| ~10 examples, getting started | `BootstrapFewShot` |
| 50+ examples, want better accuracy | `BootstrapFewShotWithRandomSearch` |
| Zero-shot, care about tone/style | `COPRO` |
| 200+ examples, extended optimization | `MIPROv2` |
| Moving to production, need efficiency | `BootstrapFinetune` |

---

### Basic Optimizer Usage Pattern

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Configure the optimizer
config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)

optimizer = BootstrapFewShotWithRandomSearch(metric=your_metric, **config)

# Compile (optimize) your program
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Save for later use
optimized_program.save("optimized_program.json")

# Load later
loaded_program = YourProgramClass()
loaded_program.load(path="optimized_program.json")
```

---

## 6. Metrics (The "Loss Function")

This is where your "Judge" lives. Metrics tell DSPy what "good" output looks like.

### Built-in Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| `dspy.evaluate.answer_exact_match` | Simple string matching | Math, code, factual QA |
| `dspy.evaluate.SemanticF1` | Checks keyword overlap | Retrieval, summarization |

### Custom Metrics

You write a Python function that returns a `float` (0.0-1.0) or `bool`.

```python
def quality_metric(example, prediction, trace=None):
    """
    Args:
        example: The input example with expected output
        prediction: The model's prediction
        trace: Optional - inspect intermediate steps during optimization

    Returns:
        bool or float (0.0 to 1.0)
    """
    # Simple boolean metric
    return prediction.answer.lower() == example.answer.lower()

def graded_metric(example, prediction, trace=None):
    """Returns a score from 0 to 1"""
    score = 0.0

    # Check for key requirements
    if "specific_term" in prediction.answer:
        score += 0.3
    if len(prediction.answer) > 50:
        score += 0.3
    if prediction.answer.endswith("."):
        score += 0.4

    return score
```

**Crucial Tip:** Pass `trace=True` to your metric to inspect the model's intermediate steps during optimization.

---

## 7. When to Use DSPy

### The Golden Rule

> **Using DSPy for a single, one-off query is like building an entire car factory just to drive to the grocery store once.**

### DSPy Requires Three Things You May Not Have

1. **Training Data:** Examples of "Good" vs. "Bad" outputs to learn from
2. **A Metric (The Judge):** Python code defining exactly what makes output "good"
3. **Scale:** Optimization requires running the prompt 20-50+ times

### Decision Framework

| Scenario | Use DSPy? | Why |
|----------|-----------|-----|
| One-off research query | **No** | No data, no scale, manual iteration faster |
| Building a product/API | **Yes** | Need reliability across thousands of inputs |
| Prototyping/exploring | **No** | Still discovering what "good" looks like |
| Production at scale | **Yes** | Can leverage data from real usage |
| Hitting accuracy ceiling | **Yes** | Optimizers find improvements you won't |

### The "App" Scenario: When DSPy Becomes Essential

Imagine building **"MarketPulse"** — an app that generates reports for *any* job title for hundreds of users:

| Feature | One-off Query | MarketPulse App (DSPy) |
|---------|---------------|------------------------|
| **Input** | Just "AI Engineer" (1 input) | "AI Engineer", "React Dev", "Nurse" (1,000+ inputs) |
| **Reliability** | If prompt fails, re-type it | If prompt fails for User #500, they churn |
| **Goal** | Get the answer *now* | Get a prompt that works *on average* for everyone |
| **Optimization** | Manual tweaking | DSPy mathematically optimizes to 95%+ success |

**DSPy's Role:** Feed it 20 examples of different roles and good reports. DSPy "compiles" a prompt robust enough to handle nuances of *all* roles, likely discovering instructions you never thought of (e.g., *"For medical roles, prioritize certification boards; for tech roles, prioritize GitHub keywords"*).

---

## 8. Decision Matrix: Choosing the Right Optimizer

### By Data Size

![Optimizer Selection by Data Size](/meta_prompting/images/Screenshot%202026-02-05%20at%2011.24.04%E2%80%AFPM.png)

**Rule of Thumb:**
- **10-50 examples:** Use `BootstrapFewShot`
- **50-200 examples:** You can safely attempt `MIPROv2` or `COPRO`
- **200+ examples:** `MIPROv2` with extended trials

---

### Detailed Optimizer Scenarios

#### A. `dspy.BootstrapFewShot` (The Workhorse)

**How it works:** Runs your "Student" model through training data. If the student gets the answer right, saves that interaction as a "Demonstration."

**Best Scenario: Cold Start / Prototyping**
- You have < 20 examples
- You are confident your prompt instructions are "okay," but model needs to see the format (e.g., JSON structure)
- **Example:** Your MarketPulse app needs to output a specific JSON schema. Bootstrap ensures perfect format by showing examples.

---

#### B. `dspy.COPRO` (Coordinate Prompt Optimization)

**How it works:** Uses an "Optimizer LLM" to propose textual improvements to your prompt prefix. Like "A/B Testing" on steroids.

**Best Scenario: Style Transfer / Nuance**
- You have no examples (Zero-shot) OR care about "Tone/Vibe"
- Model is getting facts right, but style is wrong (too robotic, too verbose)
- **Example:** You want MarketPulse reports to sound like "A Wall Street Analyst wrote it." COPRO finds magic words ("Be pithy," "Use financial vernacular").

---

#### C. `dspy.MIPROv2` (Multi-Instruction Proposal Optimizer)

**How it works:** Heavyweight Bayesian approach searching "Instruction Space" and "Example Space" simultaneously. Expensive (token-wise) but thorough.

**Best Scenario: Production / Squeeze the Last 5%**
- You have a large dataset (50+ validated examples)
- Hitting a ceiling (e.g., 92% accuracy) and simple tweaks aren't helping
- Want model to discover "Hidden Rules" you didn't think of
- **Example:** After MarketPulse runs for 2 months, you notice it fails on "Junior Roles." Feed 100 failed examples into MIPROv2, and it discovers: *"For Junior roles, prioritize 'Potential' keywords over 'Experience' keywords."*

---

## 9. Practical Examples

### Example 1: Simple Classification

```python
import dspy

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Define signature
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")

# Create module
classifier = dspy.ChainOfThought(SentimentClassifier)

# Use it
result = classifier(review="This product exceeded my expectations!")
print(result.sentiment)  # "positive"
print(result.reasoning)  # Step-by-step reasoning
```

### Example 2: RAG Pipeline

```python
class RAGPipeline(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Retrieve relevant documents
        context = self.retriever(question)

        # Generate answer with context
        return self.generate(context=context, question=question)

# Optimize it
from dspy.teleprompt import BootstrapFewShot

def answer_correct(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

optimizer = BootstrapFewShot(metric=answer_correct)
optimized_rag = optimizer.compile(RAGPipeline(my_retriever), trainset=my_data)
```

### Example 3: Agent with Tools

```python
# Define tools
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

def search(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return search_results

# Create ReAct agent
agent = dspy.ReAct(
    "question -> answer",
    tools=[calculator, search]
)

# Use it
result = agent(question="What is 15% of the population of France?")
```

---

## 10. Best Practices & Tips

### General Guidelines

1. **Start Simple:** Begin with `dspy.Predict` or `dspy.ChainOfThought` before complex modules
2. **Iterate on Signatures:** Good field names and descriptions matter more than you think
3. **Small Training Sets First:** Start with 10-20 high-quality examples before scaling
4. **Use Traces for Debugging:** Pass `trace=True` to metrics to understand failures

### Optimization Tips

1. **Bootstrap Before MIPRO:** Always try `BootstrapFewShot` before heavier optimizers
2. **Quality Over Quantity:** 20 excellent examples beat 200 mediocre ones
3. **Diverse Examples:** Include edge cases and variations in your training data
4. **Save Optimized Programs:** Always persist your optimized programs to JSON

### Cost Management

- A typical MIPROv2 optimization run costs ~$2 USD and takes ~20 minutes
- Use cheaper models for optimization, deploy with better models
- Cache aggressively during development

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Optimizing without enough data | Start with manual prompting, collect examples first |
| Overfitting to training data | Use held-out validation set |
| Ignoring the metric | Your metric IS your optimization target—make it accurate |
| Not saving optimized programs | Always `program.save()` after optimization |

---

## Resources

- **Documentation:** [dspy.ai](https://dspy.ai)
- **GitHub:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **Discord Community:** [discord.gg/XCGy2WDCQB](https://discord.gg/XCGy2WDCQB)
- **Twitter:** [@DSPyOSS](https://twitter.com/DSPyOSS)
- **Primary Paper:** "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2024)

---

*Last Updated: February 2026*
