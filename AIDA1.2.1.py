from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
import pyodbc
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
from rich.console import Console
from rich.markdown import Markdown

# Load API Keys and Credential
load_dotenv()

# For better printing
console = Console()

# Load Database
server = os.getenv("server")
database = os.getenv("database")
username = os.getenv("db_username")
password = os.getenv("password")

conn = pyodbc.connect(
     f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Cursor for using DB
db_cursor = conn.cursor()

llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0)


#Functions
def preview_tables():
    db_cursor.execute("""
                      SELECT table_name
                      FROM information_schema.tables
                      WHERE table_schema = 'dbo';                  
    """)
    tables = db_cursor.fetchall()
    st.subheader("Available Tables : \n")
    for table in tables:
        st.markdown(f"- {table[0]}")

def get_table_info(table_name):
    query = "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?"
    db_cursor.execute(query, table_name)
    table_schema = db_cursor.fetchall()
    return table_schema

def preview_data(table_name):
    query = f"SELECT TOP 10 * FROM {table_name}"
    db_cursor.execute(query)
    data_prev = db_cursor.fetchall()
    col_name = [column[0] for column in db_cursor.description] #Get Column name
    cleaned_query_data = [[np.nan if val is None else val for val in row] for row in data_prev]
    dataframe = pd.DataFrame(cleaned_query_data,columns=col_name)
    st.dataframe(dataframe)


def natural_into_query(table_name,table_schema,nl_query):
    schema_str = "\n".join([f"{col[0]} ({col[1]})" for col in table_schema])

    promt_temp = PromptTemplate.from_template("""You're a SQL expert using SSMS. Given:

                                                - Table structure: {table_structure}  
                                                - Table name: {table_name}  
                                                - User question: "{query}"  

                                                Write a valid SQL query that best answers the user's question.  
                                                - Use the column names from the structure.  
                                                - If a keyword in the question matches or refers to a column, use that column.  
                                                - Return only the SQL, no explanation.
                                              """)
    prompt = promt_temp.invoke({"table_structure" : schema_str,"table_name" : table_name,"query" : nl_query})
    sql_query = llm.invoke(prompt).content.removeprefix("```sql").removesuffix("```")
    st.write("Query Generated : ",sql_query)
    db_cursor.execute(sql_query)
    col_name = [column[0] for column in db_cursor.description] #Get Column name
    query_results = db_cursor.fetchall()
    return query_results,col_name

def get_data(data_query,column_name):
    # Cleaned Data And Turn It Into A Dataframe
    cleaned_query_data = [[np.nan if val is None else val for val in row] for row in data_query]
    results_dataframe = pd.DataFrame(cleaned_query_data, columns=column_name)
    st.dataframe(results_dataframe)
    return results_dataframe

def analyze_data(data_from_query,user_question):
    # Get User Question to Analyze DataFrame
    promt_temp = PromptTemplate.from_template("Analyze this data and provide key insight based only on the data given : \n {analyze_data} \n\n"
                                              "And Question that is asked to you : \n{question}\n"
                                              "Also provide some numeric calculation based on the data to prove this analysis. Kept the insight brief and easy to read in the terminal ")
    
    data_summary = {
        "collumn" : list(data_from_query.columns),
        "num_row" : len(data_from_query),
        "sample_data_100" : data_from_query.head(100).to_dict(orient="records"),
        "statistics" : data_from_query.describe(include ="all").to_dict()
    }

    promt = promt_temp.invoke({"analyze_data" : data_summary,"question" : user_question}) 
    insight = llm.invoke(promt).content
    st.markdown(insight)
    return insight

def suggest_chart(data_from_query,user_question,insight):
    prompt_temp = PromptTemplate.from_template(""" You are a data visualization expert. Based on the following inputs:

                                                - **Data Preview**:  
                                                {query_data}

                                                - **User Question**:  
                                                {question}

                                                - **Extracted Insight**:  
                                                {insight}

                                                Choose the **most appropriate visualization type** from the following list, and it can be 
                                                multiple chart for multiple purpose of visualizing the data, considering best practices. Respond with:

                                                1. **Chart Type** (choose one from the list below)
                                                2. **Columns Needed** (briefly state which columns to use and how)
                                                3. **Short Reason** (why this chart fits the data and insight)

                                                Available chart types:

                                                - **bar-chart**: Compare categories.  
                                                - Use grouped bars for multiple series (add `hue`).  
                                                - Use stacked bars only for 100 percent comparisons.

                                                - **line-chart**: Show trends over time.  
                                                - X-axis must be time-related (e.g., date, year).  
                                                - Multiple categories can be shown with color/line styles.

                                                - **pie-chart**: Show proportions.  
                                                - Use only if ≤5 categories.  
                                                - Total values must add up to 100%.

                                                - **scatter-chart**: Show relationships between two numeric variables.  
                                                - Add color (`hue`) for category groupings.  
                                                - Regression lines can help show trends.

                                                - **hist-chart**: Show data distribution.  
                                                - Use for numerical columns. Avoid for categorical data.  
                                                - Choose bin size wisely (e.g., `bins=20` for large datasets).

                                                - **box-chart**: Compare distributions and detect outliers.  
                                                - Best for numeric data across groups.  
                                                - Y-axis should be numeric. Highlight means if possible.

                                                - **none**: No visualization needed.
                                                Give only one best match. Keep your answer concise and clear.
                                                  """)
    prompt = prompt_temp.invoke({"query_data" : data_from_query, "question" : user_question, "insight" : insight})
    viz_suggest = llm.invoke(prompt).content.strip().lower()
    st.markdown(viz_suggest)
    return viz_suggest

def gen_visualization(data,chart_type,user_input):
    promp_temp = PromptTemplate.from_template(""" You're a Python data visualization expert using Streamlit and Plotly. Given the DataFrame:
                                                  {query_data},
                                                  Suggested chart type: {suggest_chart},
                                                  and User preference: {user_input},
                                                  Generate Python code using Plotly that:
                                                  1. Use This Template for the Code 
                                                  import plotly.express as px
                                                  import streamlit as st

                                                  df = result_data 
                                                  # your visualization code here
                                                  st.plotly_chart(fig) 

                                                  2. Prioritizes the users visualization choice (if different from the recommendation).

                                                  3. Follows best practices for professional, readable charts.

                                                  4. Uses Streamlits st.plotly_chart(fig) to render the chart.

                                                  5. Adds labels, titles, and appropriate colors for clarity.

                                                  6. Only return the Python code — no explanations.
                                              """)
    prompt = promp_temp.invoke({"query_data" : data, "suggest_chart" : chart_type,"user_input" : user_input})
    vis_gen = llm.invoke(prompt)
    gen_vis_code = vis_gen.content.removeprefix("```python").removesuffix("```")
    st.caption("AI Generated Visualization Codes")
    st.code(gen_vis_code)
    st.subheader("Visualization")
    exec(gen_vis_code)

#Streamlit UI
st.title("Analyze and Visualize Data With AI")
preview_tables()
with st.form("table_select_form"):
    table_name = st.text_input("Enter Table Name To Use : ", key="table_name_input")
    submitted = st.form_submit_button("Sumbit")
if submitted and table_name:
    table_schema = get_table_info(table_name)
    preview_data(table_name)
    with st.form("nl_query_form"):
        nl_query = st.text_input("Query : ")
        submitted_query = st.form_submit_button("Submit")
    if submitted_query and nl_query:
        st.subheader("Data Queried")
        query_result, column = natural_into_query(table_name,table_schema,nl_query)
        result_data = get_data(query_result,column)
        with st.form("analyze question form"):
            analyze_question = st.text_input("What Do You Want To Know About This Data? : ")
            submitted_question = st.form_submit_button("Submit")
        if submitted_question and analyze_question:
            st.subheader("AI Generated Insight")
            analyze_insight = analyze_data(result_data,analyze_question)
            st.subheader("AI Visualization Recommendation")
            ai_vis_recomendation = suggest_chart(result_data,analyze_question,analyze_insight)
            with st.form("vis_inquiry_form"):
                user_vis_inquiry = st.text_input("Choose Which Visualization or Request one : ")
                submitted_question_vis = st.form_submit_button("Submit")
            if submitted_question_vis and user_vis_inquiry:
                gen_visualization(result_data,ai_vis_recomendation,user_vis_inquiry)



 

    





    

