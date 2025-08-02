import os, io, re
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # ✅ OpenRouter base
    api_key= st.secrets["OPENROUTER_API_KEY"],        # ✅ OpenRouter key here
)

def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    messages = [
        {"role": "system", "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=messages,
        temperature=0.1,
        max_tokens=5  # We only need a short response
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

# === CodeGeneration TOOLS ============================================

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Set `figsize=(6,4)`, add title/labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    """

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    should_plot = QueryUnderstandingTool(query)
    col_info = {
    col: str(dtype)
    for col, dtype in df.dtypes.items()
}


    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)
    prompt += f"\n\nColumn types: {col_info}"

    messages = [{"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused."}]

# Add past user and assistant messages (summarized to fit tokens)
    for m in st.session_state.messages[-4:]:  # only the last 4 messages for context
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"][:500]})
        elif m["role"] == "assistant":
            cleaned_response = re.sub(r"<.*?>", "", m["content"])  # strip HTML
            messages.append({"role": "assistant", "content": cleaned_response[:500]})

# Add current query as user message
    messages.append({"role": "user", "content": prompt})


    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt

def GeneralizedInsightAgent(df: pd.DataFrame) -> str:
    schema = "\n".join([f"- {col}: {str(dtype)}" for col, dtype in df.dtypes.items()])
    samples = df.head(5).to_dict(orient="records")

    prompt = f"""
You are a data analyst AI. Given the following dataset:

Schema:
{schema}

Sample Data:
{samples}

Respond ONLY in markdown format with:
1. One-paragraph dataset summary.
2. 3 concise visualizations (with chart type).
3. 5 investigative questions a data scientist might explore.

Respond in a readable, structured markdown format.
"""

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=[
            {"role": "system", "content": "You are a structured data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    prompt = ReasoningCurator(query, result)
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    # Streaming LLM call
    messages = [{"role": "system", "content": "detailed thinking on. You are an insightful data analyst."}]

    for m in st.session_state.messages[-4:]:  # optional context
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"][:500]})
        elif m["role"] == "assistant":
            cleaned_response = re.sub(r"<.*?>", "", m["content"])
            messages.append({"role": "assistant", "content": cleaned_response[:500]})

    response = client.chat.completions.create(
    model="cohere/command-r-plus",
    messages=messages,
    temperature=0.2,
    max_tokens=512,
    stream=True
    )

    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token

            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    prompt = f"""
    You are a data expert. Analyze this uploaded CSV:
    - Rows: {len(df)}
    - Columns: {df.columns.tolist()}
    - Column types:
        • Numeric: {numeric_cols}
        • Categorical: {cat_cols}
        • Datetime: {datetime_cols}
    - Null values: {df.isnull().sum().to_dict()}

    Your task:
    1. Briefly describe what this dataset might represent.
    2. Suggest 3–5 exploratory visualizations that would help a user understand this dataset better.
    """
    return prompt


# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    prompt = DataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model="cohere/command-r-plus",
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# ====== chat style ========================================
def format_message(sender, message):
    if sender == "user":
        return f"""<div style="background:#1E1E1E;padding:10px;border-radius:10px;border-left:4px solid #00FFC6;color:#FFFFFF;margin-bottom:10px">
        🧑‍💻 <b>User:</b><br>{message}</div>"""
    else:
        return f"""<div style="background:#262626;padding:10px;border-radius:10px;border-left:4px solid #00FFC6;color:#CFCFCF;margin-bottom:10px">
        🤖 <b>InsightIQ:</b><br>{message}</div>"""


# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []

    left, right = st.columns([3,7])

    st.markdown("""
<style>
/* Headings */
h1, h2, h3, .st-emotion-cache-1kyxreq {
    color: #A3E4D7 !important;
    text-shadow: 0 0 2px #A3E4D7, 0 0 4px #A3E4D7;
    font-weight: bold;
}

/* Body text and paragraphs */
.st-emotion-cache-16txtl3 {
    color: #F2F2F2;
    font-size: 18px;
    font-family: 'Segoe UI', sans-serif;
}

/* Chat messages / info cards */
.stChatMessage {
    background: linear-gradient(145deg, #1A1A1D, #262626);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 0 10px rgba(163, 228, 215, 0.1);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    background-color: #A3E4D7;
    color: #1A1A1D;
    font-weight: 600;
    border: none;
    box-shadow: 0 0 10px rgba(163, 228, 215, 0.5);
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    background-color: #8FD6C2;
    box-shadow: 0 0 20px rgba(163, 228, 215, 0.8);
    transform: scale(1.05);
}

/* Text inputs */
.stTextInput > div > div > input {
    background-color: #262626;
    color: #F2F2F2;
    border: 1px solid #A3E4D7;
    border-radius: 8px;
    padding: 8px;
}

/* Optional: Subtle scroll bar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background-color: #A3E4D7;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


    with st.sidebar:
        st.title("📊 Dataset Overview")
        if "df" in st.session_state:
            st.write("**Shape:**", st.session_state.df.shape)
            st.write("**Columns:**", list(st.session_state.df.columns))
            st.write("**Nulls:**", st.session_state.df.isnull().sum().to_dict())


    with left:
        st.markdown("""
        <h1 style='
            font-family: Georgia, serif;
            font-size: 2.8rem;
            padding-bottom: 5px;
            color: #333;
        '>
            InsightIQ
        </h1>
    """, unsafe_allow_html=True)


        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        # Reset button to clear uploaded file and session state
        if st.button("🔁 Reset Upload", use_container_width=True):
            for key in ["df", "current_file", "insights", "messages", "plots"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()  # refresh the page cleanly


        if file:
        # Reset state if a new file is uploaded
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                try:
                    if file.name.endswith(".xlsx"):
                        df = pd.read_excel(file, engine="openpyxl")  # specify engine for reliability
                    else:
                        df = pd.read_csv(file, parse_dates=True)

                # Save to session state
                    st.session_state.df = df
                    st.session_state.current_file = file.name
                    st.session_state.messages = []  # clear previous chat messages
                    st.session_state.plots = []     # clear previous plots

                # Generate insights using the LLM
                    # Generate enhanced insights using the new general agent
                    with st.spinner("Generating general data insights …"):
                        st.session_state.insights = GeneralizedInsightAgent(st.session_state.df)


                except Exception as e:
                    st.error(f"❌ Failed to load file: {e}")
                    st.stop()

        # Display data preview and insights
            st.dataframe(st.session_state.df.head())
            st.markdown("### 📊 Dataset Insights")
            st.markdown(st.session_state.insights)

        else:
            st.info("Upload a CSV or Excel file to begin chatting with your data.")


    with right:
        st.markdown("""
    <h3 style='
        font-family: Georgia, serif;
        font-size: 1.5rem;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 4px;
        margin-top: 10px;
        color: #444;
    '>
        💬 Chat with Your Data
    </h3>
""", unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(format_message(msg["role"],msg["content"]), unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant responses
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                else:
                    header = f"Result: {result_obj}"

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Code accordion with proper HTML <pre><code> syntax highlighting
                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                # Combine thinking, explanation, and code accordion
                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

if __name__ == "__main__":

    main() 
