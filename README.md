# LangChain ReAct Agent with Tool-Calling

A production-ready implementation of a ReAct (Reasoning and Acting) AI Agent built using **LangChain**, local LLM integration via **Ollama (`qwen3:1.7b`)**, and tracing/observability powered by **LangSmith**.

---

## 📌 Project Overview

This repository demonstrates how to build an autonomous AI shopping assistant capable of invoking domain-specific Python tools to complete multi-step workflows.

The agent adheres strictly to system rules:
1. **Price Retrieval**: Queries catalog prices dynamically via `get_product_price`.
2. **Discount Calculation**: Applies tier-based discounts via `apply_discount` instead of estimating or using internal LLM math capabilities.
3. **Tool Chaining**: Sequentially executes tool calls based on intermediate outputs.

---

## 🛠️ Tech Stack & Architecture

- **Framework**: LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`)
- **Local LLM**: Ollama running `qwen3:1.7b`
- **Observability**: LangSmith (`@traceable`)
- **Environment Management**: `python-dotenv`

### Core Components

```text
┌─────────────────────────────────────────────────────────────┐
│                      User Request                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │                   SystemMessage Prompt                    │
 │  (Strict rules forcing tool calls & sequential execution) │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │                   LLM + bind_tools()                      │
 │      (Decides which tool to call based on context)       │
 └──────┬─────────────────────────────────────────────┬──────┘
        │                                             │
        ▼                                             ▼
┌───────────────────────┐             ┌───────────────────────┐
│   get_product_price   │             │    apply_discount     │
└───────────┬───────────┘             └───────────┬───────────┘
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │                 ReAct Loop Execution                      │
 │   (Appends ToolMessage results until completion)          │
 └───────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
.
├── lang-chain-course.py    # Main script containing agent setup & tools
├── .env                    # Environment variables (LangSmith API keys, etc.)
├── .gitignore              # Virtual environments & sensitive key ignores
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed and Ollama running locally with the target model:

```bash
ollama pull qwen3:1.7b
```

### 2. Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone <your-repository-url>
cd LANG_CHAIN_CLASS01

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
python -m pip install -U langchain langchain-core langchain-community langchain-ollama langsmith python-dotenv
```

### 3. Environment Setup

Create a `.env` file in the root directory for tracing with LangSmith (optional but recommended):

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langchain-agent-course
```

---

## 🏃 Running the Agent

Execute the main script:

```bash
python lang-chain-course.py
```

### Example Console Output

```text
Hello, LangChain Agent (.bind_tools)!

Question: What is the price of a laptop after applying a gold discount?
============================================================
    >> Executing get_product_price(product='laptop')
    >> Executing apply_discount(price=1299.99, discount_tier='gold')

[Final Answer]: The price of a laptop after applying a 23% gold discount is $1000.99.
```

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).