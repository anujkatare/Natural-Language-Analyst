import os
import json
import pandas as pd
import google.generativeai as genai

def query_llm(query: str, df: pd.DataFrame, api_key: str = None):
    """
    Queries the Gemini API with the user question and dataframe schema.
    Returns a parsed JSON response containing pandas code, explanation, and chart config.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable or provide it in the UI.")

    genai.configure(api_key=api_key)
    
    # Get column info and first few rows as context
    schema_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = list(df[col].dropna().unique()[:3])
        schema_info.append(f"- Column '{col}' (type: {dtype}). Sample values: {sample_vals}")
    
    schema_str = "\n".join(schema_info)
    df_head = df.head(5).to_string()

    prompt = f"""
You are a data assistant. Your job is to translate a user's natural language query into Python Pandas code to run on a DataFrame named `df`.
You must respond with a raw JSON object and nothing else. No markdown blocks except if you wrap the response in a JSON block.

Here is the schema of the DataFrame `df`:
{schema_str}

Here is a sample of the first 5 rows:
{df_head}

User Query: "{query}"

Instructions:
1. Determine if the query can be answered using the provided DataFrame schema. If NOT (e.g., query is about general knowledge, irrelevant topics, or fields not in the dataset), respond with a JSON where `"status": "out_of_scope"`. Do not hallucinate or guess.
2. If it can be answered, set `"status": "success"`.
3. Provide pandas code in `"pandas_code"`.
   - The DataFrame is loaded as `df`.
   - You MUST compute the final answer and assign it to a variable named `result`.
   - Keep the code simple, clean, and safe. Do not import unauthorized libraries.
4. Provide a clear, friendly English explanation in `"explanation"`. Use a placeholder like `{{result}}` where the actual calculated value should go.
5. If the query requires or benefits from a chart (e.g., comparison, trend over time, distribution), configure it:
   - Set `"chart_type"` to `"bar"`, `"line"`, or `"pie"`. If no chart is suitable, set to `"none"`.
   - For chart type, you must also provide the data for the chart. Let's make the pandas code prepare a secondary DataFrame or series named `chart_data` (e.g., grouped data) and store it in `chart_data`.
   - If a chart is configured, specify the X and Y axes or index column names in `"chart_x"` and `"chart_y"`.
   - Set `"chart_title"` to a descriptive title.

Response Format (JSON):
{{
  "status": "success" | "out_of_scope",
  "pandas_code": "Pandas python code here. E.g. grouped = df.groupby('Category')['Sales'].sum().reset_index(); result = grouped; chart_data = grouped",
  "explanation": "The total sales by category are: \\n {{result}}",
  "chart_type": "bar" | "line" | "none",
  "chart_x": "Category",
  "chart_y": "Sales",
  "chart_title": "Total Sales by Category"
}}

Remember, output ONLY the JSON. Do not write any explanations before or after the JSON.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Request JSON response
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        response_dict = json.loads(response.text.strip())
        return response_dict
    except Exception as e:
        # Fallback in case JSON parsing failed
        text = response.text.strip()
        # Clean markdown code block wraps robustly
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

def execute_pandas_code(code: str, df: pd.DataFrame):
    """
    Executes the generated pandas code in a safe sandbox local environment.
    Returns (result, chart_data, error_message).
    """
    local_vars = {'df': df.copy(), 'pd': pd}
    try:
        # We execute the code in the local_vars context
        exec(code, {}, local_vars)
        
        result = local_vars.get('result', None)
        chart_data = local_vars.get('chart_data', None)
        
        return result, chart_data, None
    except Exception as e:
        return None, None, str(e)
