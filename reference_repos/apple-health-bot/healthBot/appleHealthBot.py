from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


def classify_query(query, chat):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your job is to classify the following query into 3 categories 'workouts',  'sleep',  'exerciseTime'. "
                "You need to parse"
                "the query and classify it in to either of these 3 categories, if you are unable to classify, "
                "choose workouts by default. For exercise time, the query should ask for time spent exercising or "
                "working out, don't put every query contain the keyword time in this category, for example if the "
                "user asks how many times have they run 10k or 5k, the category should be classified as workouts. The "
                "exerciseTime table only contains data about total time spent exercising a particular day. The "
                "response content should only contain the category name and nothing else.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | chat
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=query)],
        }
    )
    print("Classified the request as ", response.content)
    return response.content.strip("'").strip("'").strip("")


def load_data_into_db(csv_path, db_path, table_name):
    engine = create_engine(db_path)
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, index=False, if_exists="replace")
    return engine


def main():
    system_prompt = """You are an agent designed to interact with a SQL database. Given an input question, create a 
    syntactically correct  query to run, then look at the results of the query and return the answer. You can order 
    the results by a relevant column to return the most interesting examples in the database.When the user refers to 
    a relative timeline such as last week, last month, compare it with the current system time. Never query for all 
    the columns from a specific table, only ask for the relevant columns given the question. You have access to tools 
    for interacting with the database. Only use the given tools. Only use the information returned by the tools to 
    construct your final answer. You MUST double check your query before executing it. If you get an error while 
    executing a query, rewrite the query and try again. The exerciseTime db contains the time in minutes, not hours, 
    make sure you do proper unit conversions before answering the query. The workouts table contains distances in 
    miles, not kilometers, make sure you do proper unit conversions when the user query contains distance in km. For 
    queries related to hiking  use HKWorkoutActivityTypeHiking, for queries related to running use 
    HKWorkoutActivityTypeRunning. Similarly, other workouts have their own identifiers in the workouts database

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer. Here is the input:
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

    # Paths to CSV files and corresponding SQLite databases
    data_paths = {
        "workouts": ("/data/export_workouts.csv", "sqlite:///workouts.db", "workouts"),
        "sleep": ("/data/export_sleep.csv", "sqlite:///sleep.db", "sleep"),
        "exerciseTime": (
            "/data/export_exercise_time.csv",
            "sqlite:///exerciseTime.db",
            "exerciseTime",
        ),
    }

    while True:
        user_input = input("Enter your query or 'exit' to quit: ")
        if user_input.lower() == "exit":
            break

        # Classify the user's query
        category = classify_query(user_input, llm)

        if category in data_paths:
            csv_path, db_path, table_name = data_paths[category]
            engine = load_data_into_db(csv_path, db_path, table_name)
        else:
            print(
                "\nAI: I'm not sure how to classify your request. Please try again with more detail.\n"
            )
            continue
        print("Processing your request!")
        db = SQLDatabase(engine=engine)
        agent_executor = create_sql_agent(
            llm, db=db, agent_type="openai-tools", verbose=False
        )

        response = agent_executor.invoke({"input": system_prompt + user_input})
        formatted_response = response.get(
            "output", "Sorry, I could not process your request."
        )
        # Print the entire response in bold and purple
        print("\033[1m\033[95mAppleHealthBot:", formatted_response, "\033[0m\n")


if __name__ == "__main__":
    main()
