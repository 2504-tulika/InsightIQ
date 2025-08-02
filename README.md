# 📊 InsightIQ - Conversational Data Analysis Assistant

InsightIQ is an AI-powered web application that allows users to **conversationally analyze any CSV dataset**. Upload your data, ask questions in natural language, get answers with **intelligent visualizations**, and explore deeper insights with **semantic memory**.

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
| File Handling   |  Auto Parsing                     |

---

## 📦 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/2504-tulika/InsightIQ
cd InsightIQ

2. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. **Install the dependencies**
```bash
pip install -r requirements.txt

4. UI Theme Configuration
Create .streamlit/config.toml and add the following:

[theme]
base = "dark"
primaryColor = "#A3E4D7"  # Soft mint/teal
backgroundColor = "#1A1A1D"  # Rich charcoal grey
secondaryBackgroundColor = "#262626"  # Muted slate grey
textColor = "#F2F2F2"  # Light grey-white (not too sharp)
font = "sans serif"


5. **Configure API Keys**
Create .streamlit/secrets.toml file and add:
OPENROUTER_API_KEY=your_key_here

5. **Run the App**
```bash
streamlit run agent.py

