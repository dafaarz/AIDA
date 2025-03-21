from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import pyodbc
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
from rich.console import Console
from rich.markdown import Markdown

# For better printing
console = Console()

# Load Database
server = "********"
database = "********"
username = "********"
password = "********"

conn = pyodbc.connect(
     f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Cursor for using DB
db_cursor = conn.cursor()

# Load API Keys 
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature = 0)

def get_table_info():
    while True:
        try:
            table_name = input("Table Name : ")
            query = "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?"
            db_cursor.execute(query,table_name)
            table_schema = db_cursor.fetchall()
            if not table_schema:
                raise ValueError(f"Theres no table name {table_name} in the database")
            else:
                return table_name,table_schema
        except ValueError as e:
            print(f"{e}. Please Try Again")
        except pyodbc.Error as e:
            print(f"{e}. Please Check Your Connection") 
            

def natural_into_query(table_name,table_schema):
    nl_query = input("Query : ")

    schema_str = "\n".join([f"{col[0]} ({col[1]})" for col in table_schema])

    promt_temp = PromptTemplate.from_template("""You're a SQL expert and using SSMS. Given the following table structure:  {table_structure}
                                              And name of the table : {table_name}. Turn this into a SQL query. Just the query, without any explanation needed : {query}
                                              """)
    prompt = promt_temp.invoke({"table_structure" : schema_str,"table_name" : table_name,"query" : nl_query})
    sql_query = llm.invoke(prompt)
    return sql_query.content.removeprefix("```sql").removesuffix("```")

def exec_query(sql_query):
    # Execute Query 
    db_cursor.execute(sql_query)
    col_name = [column[0] for column in db_cursor.description] #Get Column name
    query_results = db_cursor.fetchall()
    return query_results,col_name

def get_data(data_query,column_name):
    # Cleaned Data And Turn It Into A Dataframe
    cleaned_query_data = [[np.nan if val is None else val for val in row] for row in data_query]
    results_dataframe = pd.DataFrame(cleaned_query_data, columns=column_name)
    print(results_dataframe,)
    user_input = input("Download The Data? (Y/N) : ")
    if user_input == "Y":
        results_dataframe.to_csv("AIDA_DF.csv",index=False)
        print("Output Saved!")
        return results_dataframe
    else :  
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
   
    console.print(Markdown(f"Insight : \n {insight}"))
    user_input = input("Download Insight? (Y/N) : ")
    if user_input == "Y" :
        with open("AIDA_INSIGHT.txt", "w", encoding="utf-8") as f:
          f.write(insight)
          print("Insight Is Downloaded Into a txt file")
          return insight
    else:
        return insight

def suggest_chart(data_from_query,user_question,insight):
    prompt_temp = PromptTemplate.from_template(""" Based on the data :\n {query_data}, The user question :\n {question},
                                                  and insight that is found : {insight}. Suggest what is the best visualization  
                                                  from this list. Choose From : \n
                                                  - 'bar-chart' → for categorical comparisons.  
                                                    * Use grouped bar charts (with `hue`) for comparing categories across different groups.  
                                                    * Use stacked bars only if showing percentage distribution.  
                                               
                                                  - 'line-chart' → for trends over time.  
                                                    * Ensure x-axis has a time-based column (e.g., Date, Year).  
                                                    * If multiple categories exist, use different colors or line styles for clarity.  
                                               
                                                  - 'pie-chart' → for proportions (when categories add up to 100%).  
                                                    * Only use if there are ≤5 categories, otherwise use bar charts.  
                                                    * Ensure values sum up to 100%.  
                                               
                                                  - 'scatter-chart' → to show relationships between numeric variables.  
                                                    * Use different colors (`hue`) for categorical grouping.  
                                                    * Use regression lines if trends need highlighting.  
                                               
                                                  - 'hist-chart' → for data distribution.  
                                                    * Ensure bin size (`bins`) is well-chosen (e.g., `bins=20` for large datasets).  
                                                    * Avoid using histograms when data is categorical.  
                                               
                                                  - 'box-chart' → to detect outliers and compare distributions.  
                                                    * Use it for comparing distributions across different groups (e.g., Age by Gender).  
                                                    * Ensure numerical data is used on the y-axis.  
                                                    * Add marker for mean and important data so can be understood more quickly
                                               
                                                  - 'none' → if visualization is unnecessary.  
                                                    * also with each sugestion give what column of data needed and keep the suggestion brief
                                                  """)
    prompt = prompt_temp.invoke({"query_data" : data_from_query, "question" : user_question, "insight" : insight})
    viz_suggest = llm.invoke(prompt)
    return viz_suggest.content.strip().lower()

def gen_visualization(data,chart_type,user_input):
    promp_temp = PromptTemplate.from_template(""" You're a python data visualization master with matplotlib and seaborn given the dataframe : \n {query_data}, 
                                              ,suggested chart information : \n {suggest_chart}, and user spesification : \n {user_input}. Make a chart with the data given.
                                              Generate Python code to create the appropriate visualization and only return the code, do not include explanations.
                                              Visualization Guidelines : 
                                              - Use this template so you just focus on visualization code:
                                                import matplotlib.pyplot as plt
                                                import seaborn as sns
                                                df = data
                                                ... (vis code)

                                              - Subplots:  
                                                If multiple charts are suggested in a figure, use `plt.subplots()` to organize up to 2 chart per figure
                                                If more than 2 chart is needed, create multiple figures (pages)
                                                Use `plt.show()` to separate pages  
                                                Ensure each subplot has a **clear title and labels**.
                                              
                                              - Note for every charts 
                                                Add lables on the chart e.g for bar chart add a lable on top of the bar so it helps non technical person to see
                                                Make it as profesional as can be so the audiance can look at it better and understand it in a short glance
                                                Use apropriate color for data
                                                
                                              - Bar charts:
                                                Use `hue` for grouped bars, avoid stacking unless showing percentage breakdowns. 
                                              
                                              - Line charts:  
                                                Use *time-series data* on the x-axis.  
                                                Ensure clear distinction between multiple lines.  
                                              
                                              - Pie charts:  
                                                Use only when ≤5 categories and total sums to 100%.  
                                                Ensure labels are readable (avoid tiny slices).  
                                              
                                              - Scatter plots:  
                                                Use `hue` for categorical differentiation.  
                                                Include a regression line if showing trends. 
                                               
                                              - Histograms:
                                                Choose *optimal bin size* (`bins=20` for large datasets).  
                                                Avoid using for categorical data.  
                                              
                                              - Box plots:
                                                Use for **outlier detection** or distribution comparison.  
                                                Ensure numerical values are on the y-axis.  
                                              
                                              
                                                
                                              """)
    prompt = promp_temp.invoke({"query_data" : data, "suggest_chart" : chart_type,"user_input" : user_input})
    vis_gen = llm.invoke(prompt)
    gen_vis_code = vis_gen.content.removeprefix("```python").removesuffix("```")
    print(f"Visualization Codes : \n",gen_vis_code)
    exec(gen_vis_code)
    return

def choose_data_source():
  while True:
      try:
          choose_source = input("Choose Data Source \n 1. SQL \n 2. CSV \n >> ")
          if choose_source not in ("SQL", "CSV"):
              raise ValueError("Invalid Input. Input SSMS or CSV Only")
          else :
            return choose_source
      except ValueError as e:
          print(f"{e}. Unknown Error Try Again ") 
            
def run_AIDA(): 
  #Analyze and Visualize Data with AI
  table_name, table_schema = get_table_info()
  query = natural_into_query(table_name,table_schema)
  print(query)
  query_data,query_columns = exec_query(query)
  data = get_data(query_data,query_columns)
  input_user = input("Question About The Data? (If no Type XX): ")
  if input_user == "XX":
      print("Exiting Program!")
      sys.exit() 
  else :
      ai_insight = analyze_data(data,input_user)
      ai_vis_suggestion = suggest_chart(data,input_user,ai_insight)
      console.print(Markdown(ai_vis_suggestion))
      user_vis_suggestion = input("Any input or suggestion with the visualization given? : ")
      gen_visualization(data,ai_vis_suggestion,user_vis_suggestion)

def run_AIDA_CSV():
    data = pd.read_csv("YOUR CSV FILE NAME")
    print("Loaded CSV dataset")
    print(data.head(10))
    input_user = input("Question About The Data? (If no Type XX): ")
    if input_user == "XX":
      print("Exiting Program!")
      sys.exit() 
    else :
      ai_insight = analyze_data(data,input_user)
      ai_vis_suggestion = suggest_chart(data,input_user,ai_insight)
      console.print(Markdown(ai_vis_suggestion))
      user_vis_suggestion = input("Any input or suggestion with the visualization given? : ")
      gen_visualization(data,ai_vis_suggestion,user_vis_suggestion)    

data_source = choose_data_source()
if data_source == "SQL":
    run_AIDA()
elif data_source == "CSV":
    run_AIDA_CSV()
     






