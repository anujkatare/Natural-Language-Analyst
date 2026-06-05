import os
import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline import query_llm, execute_pandas_code

# Page settings and layout
st.set_page_config(
    page_title="AI Data Query Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium UI style styling
st.markdown("""
<style>
    /* Dark glassmorphic headers and premium font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #8A2387, #E94057, #F27121);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .refusal-card {
        background: rgba(219, 68, 85, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(219, 68, 85, 0.4);
        color: #ff4b5c;
        font-weight: 600;
        text-align: center;
        font-size: 1.3rem;
    }
    
    .success-card {
        background: rgba(46, 204, 113, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(46, 204, 113, 0.4);
        color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>AI Data Query Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions in natural language, automatically generate Pandas code, and visualize results instantly.</p>", unsafe_allow_html=True)

# Sidebar - Settings and Setup
st.sidebar.header("⚙️ Configuration")

# API Key management
api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=os.getenv("GEMINI_API_KEY", ""), 
    type="password",
    help="Enter your Gemini API key. You can also set it as GEMINI_API_KEY env variable."
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Dataset Upload")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load dataset
df = None
dataset_name = ""

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.sidebar.success(f"Loaded: {dataset_name}")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
else:
    # Use default sample data if available
    sample_path = "sample_data.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        dataset_name = "sample_data.csv (Default)"
        st.sidebar.info("Using default sample dataset.")
    else:
        st.sidebar.warning("Please upload a CSV file or verify sample_data.csv is present.")

# Main app logic
if df is not None:
    # Preview dataset in an expander
    with st.expander(f"👁️ Preview Dataset: {dataset_name} ({df.shape[0]} rows, {df.shape[1]} cols)"):
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("**Columns and types:**")
        cols_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Sample Value": [df[c].dropna().iloc[0] if not df[c].dropna().empty else None for c in df.columns]
        })
        st.table(cols_df)

    st.markdown("### 💬 Ask a Question")
    user_query = st.text_input(
        "Enter your question about the dataset:",
        placeholder="e.g., What is the total profit for each category?",
        key="query_input"
    )

    if user_query:
        if not api_key:
            st.error("⚠️ Gemini API Key is missing. Please input your key in the sidebar configuration.")
        else:
            with st.spinner("🧠 LLM is translating query to pandas code..."):
                try:
                    response_dict = query_llm(user_query, df, api_key)
                    
                    # Log generated response info
                    status = response_dict.get("status", "out_of_scope")
                    
                    if status == "out_of_scope":
                        st.markdown(
                            "<div class='refusal-card'>🚫 Out of scope: The query cannot be answered using the provided dataset.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("### 🛠️ Execution Pipeline")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("📝 Generated Python Code")
                            pandas_code = response_dict.get("pandas_code", "")
                            st.code(pandas_code, language="python")
                            
                        # Execute Code
                        result, chart_data, error = execute_pandas_code(pandas_code, df)
                        
                        with col2:
                            st.subheader("💾 Raw Output")
                            if error:
                                st.error(f"Execution Error: {error}")
                            else:
                                if isinstance(result, (pd.DataFrame, pd.Series)):
                                    st.dataframe(result, use_container_width=True)
                                else:
                                    st.metric(label="Calculated Result", value=str(result))
                        
                        # Summary Explanation
                        if not error:
                            st.subheader("💡 Answer")
                            explanation = response_dict.get("explanation", "")
                            explanation = explanation.replace("{{result}}", str(result))
                            explanation = explanation.replace("{result}", str(result))
                            st.success(explanation)
                            
                            # Chart Rendering
                            chart_type = response_dict.get("chart_type") or "none"
                            chart_type = chart_type.lower()
                            if chart_type != "none" and chart_data is not None:
                                st.subheader("📈 Visualization")
                                chart_x = response_dict.get("chart_x")
                                chart_y = response_dict.get("chart_y")
                                title = response_dict.get("chart_title", "Data Visualization")
                                
                                try:
                                    # Ensure chart_data is DataFrame
                                    if isinstance(chart_data, pd.Series):
                                        chart_data = chart_data.reset_index()
                                        
                                    if chart_type == "bar":
                                        fig = px.bar(chart_data, x=chart_x, y=chart_y, title=title, template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                    elif chart_type == "line":
                                        fig = px.line(chart_data, x=chart_x, y=chart_y, title=title, template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                    elif chart_type == "pie":
                                        fig = px.pie(chart_data, names=chart_x, values=chart_y, title=title, template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as chart_err:
                                    st.warning(f"Could not render custom chart: {chart_err}. Here is standard table view instead.")
                                    st.dataframe(chart_data)
                                    
                except Exception as e:
                    st.error(f"Error querying model or running code: {e}")
else:
    st.info("👈 Please load a dataset from the sidebar to begin.")
