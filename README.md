# 📊 InsightIQ - Conversational Data Analysis Assistant

InsightIQ is an AI-powered web application that allows users to **conversationally analyze any CSV dataset**. Upload your data, ask questions in natural language, get answers with **intelligent visualizations**, and explore deeper insights with **semantic memory** and **multi-dataset joining**.

Built using:
- 🧠 [LangChain](https://www.langchain.com/) + [OpenRouter](https://openrouter.ai/) + [Cohere Command-R-Plus]
- 🐍 Python + 🧮 Pandas + 📈 Matplotlib/Seaborn
- 🌐 Streamlit for interactive UI
- 🧠 Conversational Memory & Context Retention
- ✨ Dark-themed glowing UI for sleek user experience

---

## 🚀 Features

✅ **Natural Language Data Analysis**  
Ask questions like:  
- _“What’s the average age of passengers by survival?”_  
- _“Plot the income distribution per gender.”_

✅ **AI-Generated Code Execution**  
LLMs generate Python (pandas/matplotlib) code, executed securely on your data.

✅ **Insightful Chart Generation**  
Charts like bar plots, line graphs, pie charts, and histograms are auto-generated when helpful.

✅ **Conversational Memory**  
Your queries and results are remembered — allowing you to ask follow-up questions with context.

✅ **Multi-Table Joins**  
Upload multiple related datasets and ask questions like:  
_"Join customers.csv and orders.csv on customer_id and show total orders per customer."_

✅ **Beautiful Dark UI**  
Designed with a glowing, subtle, elegant interface for intuitive usage.

---

## 🛠️ Tech Stack

| Layer           | Technology                        |
|----------------|------------------------------------|
| Frontend        | Streamlit (Dark UI, glowing theme)|
| Backend         | Python, Pandas, Matplotlib        |
| AI Agent        | LangChain + OpenRouter (Cohere)   |
| Memory          | Streamlit Session / Custom Memory |
| File Handling   | Multi-CSV Upload + Auto Parsing   |

---

## 📦 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/InsightIQ.git
cd InsightIQ

2. **Install Dependencies**

```bash
pip install -r requirements.txt

3. **Configure API Keys**
Create a .env file in the root with:
OPENROUTER_API_KEY=your_key_here

4. **Run the App**
```bash
streamlit run agent.py

