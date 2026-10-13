import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = PROJECT_DIR / "data" / "support_knowledge.json"
RETRIEVAL_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "can", "do", "does",
    "for", "from", "have", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "the", "this", "to", "we", "what", "when", "where", "which",
    "who", "will", "with", "you", "your",
}
load_dotenv(PROJECT_DIR / ".env")


def load_knowledge(path=KNOWLEDGE_PATH):
    with path.open(encoding="utf-8") as knowledge_file:
        return json.load(knowledge_file)


def retrieve_knowledge(query, knowledge, limit=4):
    documents = [
        ("policy", f"Policy: {policy['topic']}\n{policy['details']}")
        for policy in knowledge.get("policies", [])
    ]
    documents.extend(
        (
            "intent",
            f"Intent: {intent['category']}\nExamples: {' '.join(intent['examples'])}",
        )
        for intent in knowledge.get("intents", [])
    )
    documents.extend(
        ("order", f"Sample order: {json.dumps(order, ensure_ascii=False)}")
        for order in knowledge.get("sample_orders", [])
    )

    tokenized_documents = [
        re.findall(r"[a-z0-9]+", document.lower())
        for _, document in documents
    ]
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - RETRIEVAL_STOP_WORDS
    if not query_terms or not documents:
        return ""

    exact_order_numbers = {
        order["order_number"].lower()
        for order in knowledge.get("sample_orders", [])
        if order["order_number"].lower() in query.lower()
    }
    document_frequency = Counter(
        term
        for terms in tokenized_documents
        for term in set(terms)
    )
    average_length = sum(map(len, tokenized_documents)) / len(tokenized_documents)
    ranked_documents = []

    for index, terms in enumerate(tokenized_documents):
        kind, document = documents[index]
        if (
            kind == "order"
            and exact_order_numbers
            and not any(order_number in document.lower() for order_number in exact_order_numbers)
        ):
            continue
        term_frequency = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_normalizer = 1.5 * (
                1 - 0.75 + 0.75 * len(terms) / average_length
            )
            score += (
                inverse_frequency
                * frequency
                * (1.5 + 1)
                / (frequency + length_normalizer)
            )
        if score:
            kind_weight = {"policy": 1.25, "intent": 0.75, "order": 1.5}[kind]
            ranked_documents.append((score * kind_weight, index))

    ranked_documents.sort(key=lambda result: result[0], reverse=True)
    return "\n\n".join(
        documents[index][1] for _, index in ranked_documents[:limit]
    )


def build_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Add it to your .env file."
            )
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            temperature=0.2,
        )

    raise ValueError("LLM_PROVIDER must be set to 'ollama' or 'openrouter'.")


def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the customer support assistant for {store_name}.
Use the supplied store knowledge as the only source for store-specific facts. It is fictional demo data, not live store access. Never invent order records, tracking events, policy exceptions, or completed actions. If a detail is not present, say so and ask for the minimum information needed or recommend a human support agent. Never request passwords or full payment-card details.

Classify each new customer query into exactly one category from: order_status, return_request, damaged_item, refund_status, product_question, other.
Answer helpfully and concisely, taking the conversation history into account for follow-up questions. Do not claim you have changed an order, issued a refund, or contacted a carrier.

Format every answer with this first line exactly: Category: <category>
Then put the customer-facing answer on the following lines.

Store knowledge:
{knowledge}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def to_langchain_history(messages):
    converted = []
    for message in messages:
        if message["role"] == "user":
            converted.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            converted.append(AIMessage(content=message["content"]))
    return converted


def split_category(response):
    match = re.match(r"^Category:\s*([a-z_]+)\s*\n?", response, re.IGNORECASE)
    if not match:
        return None, response
    return match.group(1).lower(), response[match.end() :].strip()


def explain_provider_error(error):
    message = str(error).lower()
    if isinstance(error, ValueError):
        return str(error)
    if "401" in message or "unauthorized" in message or "invalid api key" in message:
        return "The provider rejected the credentials. Check your API key in .env."
    if "model" in message and ("not found" in message or "does not exist" in message):
        return "The selected model was not found. Check the model name and provider setup."
    if "connection" in message or "connect" in message or "refused" in message:
        return (
            "Could not connect to the LLM provider. Check that Ollama is running, "
            "or that OpenRouter is reachable."
        )
    return "The assistant could not get a response. Check the provider settings and try again."


def render_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            category, answer = split_category(message["content"])
            if category:
                st.caption(f"Query type · {category.replace('_', ' ')}")
            st.markdown(answer or message["content"])
        else:
            st.markdown(message["content"])


def main():
    st.set_page_config(page_title="Northstar Support", page_icon="N", layout="centered")
    st.markdown(
        """
        <style>
        :root { --ink: #172a35; --teal: #146b68; --paper: #f5f7f3; }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(23, 42, 53, .10);
            border-radius: 8px;
            padding: .8rem 1rem;
            background: rgba(255, 255, 255, .78);
        }
        [data-testid="stChatInput"] { border-color: rgba(20, 107, 104, .35); }
        .support-kicker { color: var(--teal); font-size: .76rem; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    knowledge = load_knowledge()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    with st.sidebar:
        st.markdown("### Northstar")
        st.caption("Customer support desk")
        st.divider()
        st.markdown(f"**Provider**  \n`{provider}`")
        st.caption("Demo store data · no live order actions")
        if st.button("Clear conversation", icon="🗑️", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="support-kicker">NORTHSTAR MARKET · SUPPORT</div>', unsafe_allow_html=True)
    st.title("How can we help?")
    st.caption("Ask about an order, return, damaged item, or refund.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info(
            "This assistant uses fictional sample policies and orders. "
            "Replace `data/support_knowledge.json` before using it for a real store."
        )

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Write a message")
    if question:
        previous_messages = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        render_message(st.session_state.messages[-1])

        try:
            chain = build_chain(build_llm())
            retrieval_query = " ".join(
                [
                    *(
                        message["content"]
                        for message in previous_messages[-4:]
                        if message["role"] == "user"
                    ),
                    question,
                ]
            )
            retrieved_knowledge = retrieve_knowledge(retrieval_query, knowledge)
            with st.chat_message("assistant"):
                with st.spinner("Checking the support information..."):
                    response = chain.invoke(
                        {
                            "store_name": knowledge["store_name"],
                            "knowledge": retrieved_knowledge
                            or "No directly relevant information was found in the support knowledge base.",
                            "history": to_langchain_history(previous_messages),
                            "input": question,
                        }
                    )
                category, answer = split_category(response)
                if category:
                    st.caption(f"Query type · {category.replace('_', ' ')}")
                st.markdown(answer or response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except Exception as error:
            st.error(explain_provider_error(error))


if __name__ == "__main__":
    main()