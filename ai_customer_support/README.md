# AI Customer Support Assistant

A Streamlit chatbot for e-commerce support, built with LangChain. It classifies each question, uses prior messages for follow-ups, and answers from a small JSON knowledge base. The included policies and orders are fictional demo data, not a trained model or a connection to a real store.

## Requirements

- Python 3.10 or newer
- One LLM provider: Ollama running locally, or an OpenRouter API key

## Install

From this directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Choose a provider

Copy `.env.example` to `.env`. The default provider is Ollama:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running, then leave `LLM_PROVIDER=ollama` in `.env`. To use another installed model, update `OLLAMA_MODEL`.

For OpenRouter, set these values in `.env` instead:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Never commit `.env`; it is excluded by `.gitignore`.

## Run

```bash
streamlit run app.py
```

Streamlit opens the chat UI in your browser. Use **Clear conversation** in the sidebar to reset the current session. Conversation history is kept for follow-up questions during that browser session.

## Data and evaluation

Edit `data/support_knowledge.json` to replace the example policies, intent examples, and sample orders. Keep only approved, non-sensitive information in this file. The prompt asks the LLM to use this context, label one of six query categories, and avoid claiming it performed actions. This is prompt-based classification, not model training or guaranteed retrieval accuracy.

Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

Try delayed-order, damaged-item, return, and refund questions in the UI. The order examples `NS-1048` and `NS-2091` are the only sample records; the assistant should say when it has no matching record. Review responses against your approved policies before using the app with real customers.