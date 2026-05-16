# LLM Evaluator

A simple Python-based project for testing and evaluating LLM responses with a focus on grounding.
The current version is intentionally minimal. It uses a Groq-backed adapter to generate responses and a grounding metric to check whether the response is supported by the provided context.

## Project goal
The purpose of this project is to build a lightweight evaluator that is easy to understand, extend, and maintain.
Instead of adding many complex validation rules early, the project currently focuses on one core idea:
	•	If a model response is supported by the supplied context, it should score well on grounding.
	•	If no supporting context is provided, the response should not be considered grounded, even if it is factually correct.
This follows common LLM evaluation practice where groundedness measures support from context rather than general world knowledge.

## Current architecture
The project is organized around a few small components:
	•	 GroqLLMAdapter  — sends prompts to Groq models and returns generated text.
	•	 LLMTestCase  — stores the input, expected output, actual output, and context.
	•	 GroundingMetric  — scores whether the generated output is supported by the provided context.
	•	Small runner scripts — execute one or more test cases and print or export results.
This keeps the framework modular while staying simple enough for fast iteration and debugging.

## Setup

### 1. Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Add environment variables
Create a .env file:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Suggested Folder Structure
```
llm_evaluator/
├── llm_tester/
│   ├── adapters/
│   │   ├── base.py
│   │   └── groq_adapter.py
│   ├── metrics/
│   │   └── grounding.py
│   ├── models/
│   │   └── test_case.py
│   └── loaders/
│       └── json_loader.py
├── data/
│   └── cases_groq.json
├── run_groq_single.py
├── run_groq_evaluator.py
├── .env
├── requirements.txt
└── README.md

```

## Running Tests
Example:
```
python run_groq_single.py
```
Typical flow:
	1.	Send a prompt to Groq.
	2.	Receive the generated answer.
	3.	Build an  LLMTestCase .
	4.	Score it with  GroundingMetric .
	5.	Print the result.

## Key concept: grounding vs correctness
This project currently focuses on grounding, not full factual correctness.
That means:
	•	A response can be factually correct but still get grounding  0.0  if the context is empty.
	•	A response should score highly only when the provided context supports it.
This distinction is important in RAG-style and context-based evaluation workflows.

## Switching Groq models
In Code:
```
adapter = GroqLLMAdapter(model="llama-3.3-70b-versatile")
```

## Recommended next steps
The simplest roadmap for this project is:
	•	Keep the grounding evaluator small and reliable.
	•	Add a few grounded test cases with explicit context.
	•	Store model outputs in CSV for inspection.
	•	Compare different Groq models on the same grounded prompts.
	•	Add more metrics only when there is a clear need.
  
This staged approach aligns with practical LLM evaluation guidance, which recommends starting with clear, task-relevant metrics instead of overloading the framework too early.
