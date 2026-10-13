# E-commerce Customer Support Assistant

## Run from the repository root

Install the app requirements into the Python environment used by `streamlit`:

```bash
python -m pip install -r ai_customer_support/requirements.txt
```

The app uses Ollama by default. Start its service in a separate terminal and
leave it running:

```bash
ollama serve
```

In another terminal, download the configured model:

```bash
ollama pull llama3.2
```

Then start Streamlit from this repository root:

```bash
streamlit run app.py
```

Alternatively, follow [the app README](ai_customer_support/README.md) to run
the app from its own directory or configure OpenRouter.

Chat responses retrieve relevant policies, intent examples, and sample orders
from `ai_customer_support/data/support_knowledge.json` before sending the
customer's question to the language model.