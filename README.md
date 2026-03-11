# 🤖 ArxivistaResearchAgent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arxivista.streamlit.app/)

Welcome to **Arxivista**, your intelligent AI-powered research assistant built with [Streamlit](https://streamlit.io), [LangGraph](https://github.com/langchain-ai/langgraph), and [LangChain](https://www.langchain.com/). It leverages a multi-step agent to fetch, process, and synthesize high-quality academic and web-based knowledge — all in one place.

---

## 🚀 Launch the Web App

👉 Click below to open the app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arxivista-research-agent.streamlit.app/)

---

## 🔍 What Can Arxivista Do?

Arxivista helps you:

- 📚 **Fetch research papers** directly from [arXiv.org](https://arxiv.org/) based on your selected category and number of results.
- 📄 **Download, process, and chunk PDFs**, creating vector embeddings using OpenAI.
- 📦 **Index documents into Pinecone** for efficient and fast semantic search.
- 🔎 **Ask natural language questions** and receive:
  - 🤖 Retrieval-augmented answers using relevant academic content (RAG)
  - 🌐 Web search results via SerpAPI (when knowledge base lacks coverage)
  - 📄 Paper-specific summaries using ArXiv IDs
- 🧠 **Combine multiple tools** dynamically (LangGraph agent decides the best toolchain).
- 📊 Return a **structured research report** with citations, summaries, and sources.

---

## 🧠 How It Works

The app uses a **graph-based agent** with decision-making capabilities that:

1. Takes your query as input.
2. Chooses from a set of tools:
   - ArXiv fetcher
   - Web searcher (SerpAPI)
   - RAG search (with or without filters)
   - Final answer formatter
3. Generates a complete report including:
   - Introduction
   - Research steps
   - Main body
   - Conclusion
   - Sources

---

## 🛠 Built With

- [Streamlit](https://streamlit.io/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://www.langchain.com/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [OpenAI Embeddings](https://platform.openai.com/)
- [SerpAPI](https://serpapi.com/)

---

## 📁 Repository Structure

```bash
project/
├── app.py
├── pages/
│   ├── 1_Configure_ArXiv.py
│   ├── 2_Build_Knowledge.py
│   └── 3_Ask_Research_Agent.py
├── src/
│   ├── tools/
│   ├── data/
│   ├── decision/
│   └── config.py
└── .env
