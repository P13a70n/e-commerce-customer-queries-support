# E-commerce Customer Support Assistant

## Run from the repository root

Install the app requirements into the Python environment used by `streamlit`:

```bash
python -m pip install -r ai_customer_support/requirements.txt
```

Set a Google Gemini API key in `ai_customer_support/.env`. If that file does
not already exist, create it from the sample:

```bash
cp ai_customer_support/.env.example ai_customer_support/.env
```

Set `GEMINI_API_KEY` in that `.env` file. The default model is
`gemini-3.8-flash`.

Then start Streamlit from this repository root:

```bash
streamlit run app.py
```

Alternatively, follow [the app README](ai_customer_support/README.md) to run
the app from its own directory.

Chat uses BM25 retrieval over
`ai_customer_support/data/support_knowledge.csv` and sends relevant rows to
Gemini to ground its response.