# AI Customer Support Assistant

A Streamlit chatbot for e-commerce support, built with LangChain and Google Gemini. It classifies each question, uses prior messages for follow-ups, and retrieves relevant policies, intent examples, and sample orders from a CSV knowledge base. The included policies and orders are fictional demo data, not a trained model or a connection to a real store.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Install

From this directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Configure Gemini

Copy `.env.example` to `.env` and set the Gemini API key:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. The default model is `gemini-3.8-flash`; set
`GEMINI_MODEL` to select another Gemini model.

Never commit `.env`; it is excluded by `.gitignore`.

## Run

```bash
streamlit run app.py
```

Streamlit opens the chat UI in your browser. Use **Clear conversation** in the sidebar to reset the current session. Conversation history is kept for follow-up questions during that browser session.

## Data and evaluation

Edit `data/support_knowledge.csv` to replace the example policies, intent examples, and sample orders. Keep its `type`, `title`, and `content` columns, and only include approved, non-sensitive information. Each chat turn ranks matching rows with BM25 text relevance and sends up to four retrieved entries, along with the conversation, to Gemini. This file-based retrieval-augmented generation (RAG) grounds Gemini's answers in the supplied store information; it does not train or fine-tune the model. The prompt also asks Gemini to label one of six query categories and avoid claiming it performed actions. Retrieval and classification are not guaranteed to be accurate.

Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

Try delayed-order, damaged-item, return, and refund questions in the UI. The order examples `NS-1048` and `NS-2091` are the only sample records; the assistant should say when it has no matching record. Review responses against your approved policies before using the app with real customers.