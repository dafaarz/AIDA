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

def analyze_data(data_from_query,user_question,previous_insight = None):
    # Get User Question to Analyze DataFrame
    promt_temp = PromptTemplate.from_template("Analyze this data and provide key insight based only on the data given : \n {analyze_data} \n\n"
                                              "And Question that is asked to you : \n{question}\n")
    
    if previous_insight:
        promt_temp += ("Previous insight generated : {prev_insight} \n "
                       "User was not satisfied with the results generated above, refine or add new insight based on the feedback (Question) above \n")

    promt_temp += ("Also provide numeric calculations based on the data to prove this analysis. Keep it brief and easy to read")    
    
    data_summary = {
        "collumn" : list(data_from_query.columns),
        "num_row" : len(data_from_query),
        "sample_data_100" : data_from_query.head(100).to_dict(orient="records"),
        "statistics" : data_from_query.describe(include ="all").to_dict()
    }

    if previous_insight :
        promt = promt_temp.invoke({"analyze_data" : data_summary,
                                "question" : user_question,
                                "prev_insight" : previous_insight or ""})     
    else :
        promt = promt_temp.invoke({"analyze_data" : data_summary,"question" : user_question})   
    insight = llm.invoke(promt).content
    #st.markdown(insight)
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
#Form 1 (Select Tables)
if "selected_table" not in st.session_state:
    with st.form("table_select_form"):
        table_name = st.text_input("Enter Table Name To Use : ", key="table_name_input")
        submitted = st.form_submit_button("Sumbit")
        if submitted and table_name:
            st.session_state["selected_table"] = table_name
if "selected_table" in st.session_state:
    table_name = st.session_state["selected_table"]
    table_schema = get_table_info(table_name)
    preview_data(table_name)

    #Form 2 (Natural Language to SQL)
    if "nl_query_result" not in st.session_state:
        with st.form("nl_query_form"):
            nl_query = st.text_input("Query : ")
            submitted_query = st.form_submit_button("Submit")
        if submitted_query and nl_query:
            query_result, column = natural_into_query(table_name,table_schema,nl_query)
            st.session_state["nl_query_result"] = query_result
            st.session_state["nl_column"] = column
    if "nl_query_result" in st.session_state:
        st.subheader("Data Queried")
        result_data = get_data(st.session_state["nl_query_result"],st.session_state["nl_column"])
        if "last_insight" not in st.session_state:
            with st.form("analyze question form"):
                analyze_question = st.text_input("What Do You Want To Know About This Data? : ",key="analyze_question_input")
                submitted_question = st.form_submit_button("Analyze")
            if submitted_question and analyze_question:
                analyze_insight = analyze_data(result_data, analyze_question)
                st.session_state["last_insight"] = analyze_insight
                st.session_state["analyze_question"] = analyze_question

        if "last_insight" in st.session_state:
            st.subheader("AI Generated Insight")
            st.markdown(st.session_state["last_insight"]) #Always Show Past Results

            #Feedback Loop 
            if "last_insight" in st.session_state:
                # Ask For Feedback
                if "feedback_given" not in st.session_state:
                    st.write("Satisfied With The Results?")
                    col1,col2 = st.columns(2)
                    with col1:
                        if st.button("Yes"):
                            st.session_state["feedback_given"] = "Yes"
                    with col2:
                        if st.button("No"):
                            st.session_state["feedback_given"] = "No"

                # 2A If no ask for clarifications 
                if st.session_state.get("feedback_given") == "No" :   # and "feedback_question" not in st.session_state:
                    with st.form("feedback_form"):
                        feedback_question = st.text_input("Enter your clarification or new question",key="feedback_question")
                        submitted_feedback = st.form_submit_button("Iterate")
                        if feedback_question and submitted_feedback:
                            new_insight = analyze_data(result_data,feedback_question,st.session_state["last_insight"])
                            st.markdown(new_insight)
                            st.session_state["last_insight"] = new_insight  # update the latest
                            st.session_state["feedback_question_user"] = feedback_question
                            st.session_state["feedback_given"] = "Yes"

                # 2B if Yes the go to Vis
                if st.session_state.get("feedback_given") == "Yes" and "ai_vis_recomendation" not in st.session_state:
                    st.subheader("AI Visualization Recommendation")
                    ai_vis_recomendation = suggest_chart(result_data,
                                            st.session_state.get("feedback_question_user", st.session_state["analyze_question"]),
                                            st.session_state["last_insight"])
                    with st.form("vis_inquiry_form"):
                        user_vis_inquiry = st.text_input("Choose Which Visualization or Request one : ")
                        submitted_question_vis = st.form_submit_button("Submit")
                    if submitted_question_vis and user_vis_inquiry:
                        gen_visualization(result_data,ai_vis_recomendation,user_vis_inquiry)



 

    





    

