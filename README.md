# WikipediaAIAgent

This repository contains a Wikipedia AI Agent built from scratch using Llama 3 70B to answer user questions by combining reasoning with Wikipedia's knowledge base. The agent uses custom “tools” to handle both mathematical calculations and Wikipedia-specific queries, providing concise and contextually rich answers for both simple and complex queries.

[Try it out here](https://huggingface.co/spaces/AseemD/WikipediaAgent)

![image](images/app_demo.gif)

## General Overview

Wikipedia Agent is an AI chatbot that utilizes a large language model (LLM) trained on Llama 3 70B to provide accurate and context-rich answers. When faced with queries that demand detailed information:

1. It first retrieves brief information (summaries) from Wikipedia.
2. If the summary alone is insufficient to answer the question, it fetches the entire Wikipedia page, ingests it into a Vector Database, and performs a semantic similarity search to extract only the most relevant content.
3. The agent then uses the top-k similar documents combined with its own reasoning to provide a thorough, contextually-grounded answer.
4. If a calculation is required, the agent uses a calculator tool to perform the calculation and returns the result.

## Examples

1. **Basic Query**

* User: “Who invented the telephone?”
* Agent:
    * Calls wikipedia_search() with “telephone.”
    * Returns a quick summary about Alexander Graham Bell.
    * Agent responds:

        “Alexander Graham Bell is credited with the invention of the telephone…”

2. **Complex Query**

* User: “Explain how quantum entanglement is demonstrated in the EPR paradox and reference any experiments.”
* Agent:
    * Tries wikipedia_search() with “quantum entanglement,” but the summary might be too short.
    * Fetches full Wikipedia pages on “Quantum Entanglement” and “EPR paradox,” stores in VectorDB.
    * Performs semantic similarity to find relevant sections on experiments demonstrating EPR.
    * Provides a detailed explanation with references to John Bell’s experiments, etc.

3. Math-Integrated Query

* User: “Calculate the area of a circle with radius 10 plus 150.”
* Agent:
    * Interprets the request as “π * (10^2) + 150” and calls calculate().
        Returns the computed result.

## Repositiry Strcture:

```
.
├── images
│   └── ...       
│
├── utils
│   └── context.py         # Contains the system prompt for the agent
│
├── agent.ipynb            # Jupyter notebook for experimentation and prototyping
├── agent.py               # Core agent logic (LLM interactions, tool usage, etc.)
├── app.py                 # Gradio app for user interaction
└── README.md              # Project documentation
```