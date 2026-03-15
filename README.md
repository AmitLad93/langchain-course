# Structured Case Analysis with LangChain and LangSmith

This project demonstrates a **production-style LLM workflow** for converting unstructured case notes into **validated structured data** using:

* **LangChain** for model orchestration
* **Pydantic** for structured output validation
* **LangSmith** for tracing and observability
* **dotenv** for secure environment configuration

The system analyses free-text case notes and produces a **structured analysis object** containing summaries, impact points, risk indicators, timelines, and recommended next actions.

The design emphasises **reliability, observability, and modular architecture**, which are critical when building real-world AI systems.

---

# Overview

Large language models are powerful but produce **unstructured outputs by default**.
This project demonstrates how to convert LLM responses into **structured, validated Python objects** suitable for downstream systems such as:

* dashboards
* triage systems
* analytics pipelines
* quality assurance tools
* operational reporting

The workflow is designed using **LangChain's LCEL pipeline architecture** and includes **LangSmith tracing** to enable debugging and inspection of model behaviour.

---

# Architecture

The application pipeline follows a clear structure:

```
Prompt Template
      ↓
Chat Model
      ↓
Structured Output Parser (Pydantic)
      ↓
Validated Python Object
```

This architecture ensures:

* prompts remain **separate from application logic**
* outputs are **validated before use**
* pipelines remain **composable and extensible**

---

# Key Features

## Structured LLM Outputs

The system uses **Pydantic schemas** to enforce output structure.

Benefits:

* prevents unpredictable free-text responses
* ensures schema validation
* allows safe downstream processing

Example schema fields:

* customer issue summary
* product area
* issue category
* severity level
* customer impact points
* vulnerability indicators
* risk flags
* timeline events
* recommended actions
* follow-up questions

---

## Provider-Agnostic Model Initialization

Models are initialized using:

```python
init_chat_model()
```

This allows the underlying provider to be swapped without rewriting application logic.

Supported providers include:

* OpenAI
* Anthropic
* Google
* local models via Ollama

---

## Prompt Design

Prompts are constructed using:

```python
ChatPromptTemplate.from_messages()
```

This provides:

* clear separation of **system instructions**
* explicit **user input templating**
* compatibility with chat-based LLM APIs

---

## LangSmith Observability

The project integrates **LangSmith tracing** using the `@traceable` decorator.

Tracing provides:

* full visibility into prompt inputs
* model outputs
* pipeline execution order
* debugging of failures

This is essential for **production LLM systems**, where understanding model behaviour is critical.

---

# Example Output

Running the script produces structured analysis like:

```
CASE ANALYSIS RESULT

SUMMARY
Customer reports account lock preventing payments for rent and utilities.

PRODUCT AREA
Account access

ISSUE CATEGORY
Payment failure

SEVERITY
High

CUSTOMER IMPACT
- Potential late rent payment
- Financial stress

VULNERABILITY INDICATORS
- Financial stress

SERVICE RISK FLAGS
- Conflicting support guidance

TIMELINE EVENTS
- Multiple declined payments
- Customer contacted support twice
- Issue unresolved after 3 days

RECOMMENDED ACTIONS
- Review fraud trigger
- Restore account access
- Provide clear explanation to customer

FOLLOW-UP QUESTIONS
1. When did the payment attempts occur?
2. Were any transactions flagged as fraudulent?
```

---

# Design Principles

This project demonstrates several best practices when building LLM applications:

## Deterministic Outputs

Model temperature is set to **0** to reduce randomness and increase reproducibility.

---

## Fail-Fast Configuration Validation

Environment variables are validated before execution to prevent runtime failures.

---

## Modular Architecture

Responsibilities are separated into clear components:

* prompt construction
* model initialization
* pipeline assembly
* execution workflow
* output formatting

---

## Observability

LangSmith tracing ensures the behaviour of the LLM pipeline can be inspected and debugged.

---

# Technologies Used

* Python
* LangChain
* LangSmith
* Pydantic
* python-dotenv
