from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_token = os.getenv('TODOIST_API_KEY')
gemini_token = os.getenv('GEMINI_API_KEY')


todoist = TodoistAPI(todoist_token)
@tool #does something specific
def add_task(task,description=None):
    """  Add a new task to todoist"""
    todoist.add_task(content=task,
                     description=description)

@tool
def show_tasks():

    """ Show only the tasks you added (ignore Todoist default tasks).  """
    results_paginator = todoist.get_tasks()
    tasks = []

    ignore_tasks = [
        "Add your first task",
        "Download Todoist apps",
        "Add your first Project",
        "Subscribe for monthly productivity inspiration",
        "Check off tasks as you complete them",
        "Connect your calendar",
        "Add Todoist to your email",
        "Explore our curated templates",
        "Set aside 5 minutes to review your `Inbox`",
        "Go to your `Upcoming` or `Today` views",
        "Add tasks as soon as they come to mind 💡",
        "Watch: Switching to Todoist 📽️",
        "[Download Todoist apps](https://www.todoist.com/downloads) 📱💻",
        "Go to your `Upcoming` or `Today` views to see what's on your plate",
        "[Connect your calendar](https://app.todoist.com/app/settings/calendars)",
        "Go to your `Upcoming` or `Today` views to see what's on your plate",
        "[Add Todoist to your email](https://www.todoist.com/integrations/category/email)",
        "Type **`q`** to add tasks from any screen in the desktop app",
        "[Subscribe for monthly productivity inspiration](https://www.todoist.com/inspiration) 💌",
        "Explore our [curated templates](https://app.todoist.com/app/templates/category/my-templates)",
        "Check off tasks as you complete them! 🎉",

    ]

    for task_list in results_paginator:
        for task in task_list:
            if task.content not in ignore_tasks:
                tasks.append(task.content)

    if not tasks:
        return "No personal tasks try finding some"
    return "\n".join([f" - {tasks}" for task in tasks])





tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    google_api_key = gemini_token,
    temperature=0.3 #the lower less creative
)

system_prompt = """You are a helpful assistant.
 You will help the user add tasks.
 You will help users show existing tasks. If the user asks to show the tasks: for example "show me the tasks" print out the taks to the user. Print them in a bullet list format
 """


prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user","{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=False)


# response = chain.invoke({"input": user_input})



history = []
while True:
    user_input = input("You:  ")
    response = agent_executor.invoke({"input": user_input,"history":history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))