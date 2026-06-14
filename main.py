from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydanticField
from typing import Optional
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
from pathlib import Path
from database import engine, create_db_and_tables
import os

from dotenv import load_dotenv
from openai import OpenAI

import json

import logging

from experiments.rag_local.query_chroma import search_chroma
from experiments.rag_local.query_rag_chroma import ask_rag

from routers.rag import router as rag_router
from routers.tickets import router as tickets_router
from routers.agent_ticket import router as agent_ticket_router
from routers.agent_ops import router as agent_ops_router

from models.ticket import Ticket  # noqa: F401
from models.agent_ops import AgentRun, ApprovalRequest, ToolCall  # noqa: F401


load_dotenv()   # 从 .env 文件加载环境变量


class Settings(BaseModel):
    dashscope_api_key: Optional[str] = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
    dashscope_model: str = "qwen3.5-plus"
    database_url: str = "sqlite:///data/todos.db"
    sql_echo: bool = True


def get_settings() -> Settings:
    return Settings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        dashscope_model=os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/todos.db"),
        sql_echo=os.getenv("SQL_ECHO", "true").lower() == "true",
    )


settings = get_settings()   

logging.basicConfig(
    level=logging.INFO, # 设置日志级别为 INFO，意味着 INFO 级别及以上的日志都会被记录下来
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",  # 日志格式：时间戳 | 日志级别 | 记录器名称 | 日志消息
)

logger = logging.getLogger("ai-todo-api")   # 创建一个名为 "ai-todo-api" 的日志记录器实例，可以在代码中使用 logger.info(), logger.error() 等方法来记录日志消息。


llm_client = OpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)


# 定义一个 SQLModel 模型，表示数据库中的 "todo" 表格结构
class Todo(SQLModel, table=True):   #　table=True 表示这是一个数据库表格模型，表名默认为 "todo"
    id: Optional[int] = SQLField(default=None, primary_key=True)    # id可以一开始为空，它是主键。
    title: str
    completed: bool = False
    due_time: Optional[str] = None  # 任务的截止时间

# 生命周期函数，在 FastAPI 启动时自动调用
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield   # yield 前面的代码：应用启动时执行。后面的代码：应用关闭时执行。


app = FastAPI(title="FastAPI Todo API", lifespan= lifespan) # 把 lifespan 传给 FastAPI 实例，应用启动时自动调用 create_db_and_tables() 来创建数据库表格。

app.include_router(rag_router)  
app.include_router(tickets_router)
app.include_router(agent_ticket_router)
app.include_router(agent_ops_router)

""" todos = [
    {
        "id": 1,
        "title": "Learn FastAPI",
        "completed": False,
    },
    {
        "id": 2,
        "title": "Build my first Todo API",
        "completed": False,
    },
]

next_id = 3
 """

class TodoCreate(BaseModel):
    title: str = PydanticField(..., min_length=1, max_length=100)
    due_time: Optional[str] = PydanticField(default=None, max_length=100)

class TodoUpdate(BaseModel):
    title: Optional[str] = PydanticField(default=None, min_length=1, max_length=100)
    completed: Optional[bool] = None
    due_time: Optional[str] = PydanticField(default=None, max_length=100)

# 从数据库中查询出来的 Todo 记录，返回给用户的响应格式
class TodoResponse(BaseModel):
    id: int
    title: str
    completed: bool
    due_time: Optional[str] = None

# 用户输入的一段消息，要求 AI 模型原样重复输出
class EchoRequest(BaseModel):
    message: str = PydanticField(..., min_length=1, max_length=200)
    repeat: int = PydanticField(1, ge=1, le=5)

# 用户输入的一段消息，发送给 AI 模型进行对话
class ChatRequest(BaseModel):
    message: str = PydanticField(..., min_length=1, max_length=500)

#　用户输入的一段自然语言
class TaskExtractRequest(BaseModel):   
    text: str = PydanticField(..., min_length=1, max_length=1000)

# 从用户输入的自然语言中提取出的任务信息
class ExtractedTask(BaseModel):
    title: str
    time: Optional[str] = None

# 从用户输入的自然语言中提取出的任务信息列表
class TaskExtractResponse(BaseModel):
    tasks: list[ExtractedTask]

# 从用户输入的自然语言中提取出任务信息，并在数据库中创建对应的 Todo 记录，返回给用户
class AICreateTodosResponse(BaseModel):
    extracted_tasks: list[ExtractedTask]    
    created_todos: list[TodoResponse]

'''
从 LLM 返回的文本中提取出 JSON 数据，进行清洗和解析
'''
def parse_json_from_llm(text: str) -> dict:

    cleaned = text.strip()  # 去掉文本前后的空白字符

    if cleaned.startswith("```json"):   # 如果文本以 ```json 开头，去掉这个前缀
        cleaned = cleaned.removeprefix("```json").strip()

    if cleaned.startswith("```"):   # 如果文本以 ``` 开头，去掉这个前缀
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"): # 如果文本以 ``` 结尾，去掉这个后缀
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}


@app.get("/health")
def check_health():
    return {
        "status": "ok",
        "time": datetime.now().isoformat()
    }


@app.get("/about")
def about():
    return {
        "name": "fastapi-todo-api",
        "version": "0.1.0",
        "description": "My first FastAPI backend project"
    }


@app.post("/echo")
def echo(request: EchoRequest):
    return {
        "you_said": request.message,
        "repeat": request.repeat,
        "result": [request.message] * request.repeat
    }


@app.post("/chat")
def chat(request: ChatRequest):
    fake_response = f"这是一个模拟 AI 回复：我收到了你的消息「{request.message}」"

    return {
        "user_message": request.message,
        "assistant_message": fake_response
    }

@app.get("/todos", response_model=list[TodoResponse])
def list_todos():
    with Session(engine) as session:
        statement = select(Todo)
        todos = session.exec(statement).all()
        return todos


@app.post("/todos", response_model=TodoResponse, status_code=201)
def create_todo(todo: TodoCreate):
    logger.info("Creating todo: %s", todo.title)
    
    db_todo = Todo(
        title=todo.title,
        completed=False,
        due_time=todo.due_time,
    )

    with Session(engine) as session:
        session.add(db_todo)
        session.commit()
        session.refresh(db_todo)
        return db_todo


@app.get("/todos/{todo_id}", response_model=TodoResponse)
def get_todo(todo_id: int):
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)   # session.get() 方法根据主键（这里是 id）查询数据库中的 Todo 记录

        if todo is None:
            raise HTTPException(status_code=404, detail="Todo not found")

        return todo


@app.patch("/todos/{todo_id}", response_model=TodoResponse)
def update_todo(todo_id: int, update: TodoUpdate):
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)   

        if todo is None:
            raise HTTPException(status_code=404, detail="Todo not found")

        if update.title is not None:
            todo.title = update.title

        if update.completed is not None:
            todo.completed = update.completed
        
        if update.due_time is not None:
            todo.due_time = update.due_time

        session.add(todo)
        session.commit()
        session.refresh(todo)

        return todo


@app.delete("/todos/{todo_id}", status_code=200)
def delete_todo(todo_id: int):
    logger.info("Deleting todo with ID: %d", todo_id)
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)

        if todo is None:
            raise HTTPException(status_code=404, detail="Todo not found")

        session.delete(todo)
        session.commit()

        return {"message": "Todo deleted"}

# 用户输入的一段消息，发送给 AI 模型进行对话
@app.post("/ai/chat")
def ai_chat(request: ChatRequest):
    logger.info("Calling LLM chat endpoint with model: %s", settings.dashscope_model)

    if not settings.dashscope_api_key:
        raise HTTPException(
            status_code=500,
            detail="DASHSCOPE_API_KEY is not configured",
        )

    try:
        completion = llm_client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful programming tutor.",
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
        )

        assistant_message = completion.choices[0].message.content

        return {
            "user_message": request.message,
            "assistant_message": assistant_message,
            "model": settings.dashscope_model,
        }

    except Exception as e:
        logger.exception("LLM chat failed")
        raise HTTPException(
            status_code=500,
            detail=f"LLM API error: {str(e)}",
        )

# 从用户输入的自然语言中提取出任务信息，返回给用户 
@app.post("/ai/extract-tasks", response_model=TaskExtractResponse)
def extract_tasks(request: TaskExtractRequest):
    return extract_tasks_from_text(request.text)

def extract_tasks_from_text(text: str) -> TaskExtractResponse:
    if not settings.dashscope_api_key:
        raise HTTPException(
            status_code=500,
            detail="DASHSCOPE_API_KEY is not configured",
        )

    try:
        completion = llm_client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an information extraction assistant. "
                        "Extract todo tasks from the user's text. "
                        "Return ONLY valid JSON. "
                        "Do not include markdown. "
                        "The JSON format must be: "
                        '{"tasks":[{"title":"string","time":"string or null"}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        raw_content = completion.choices[0].message.content
        data = parse_json_from_llm(raw_content)

        return TaskExtractResponse(**data)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM API error: {str(e)}",
        )
    
@app.post("/ai/create-todos", response_model=AICreateTodosResponse, status_code=201)
def ai_create_todos(request: TaskExtractRequest):
    logger.info("Creating todos from AI extraction")

    extracted = extract_tasks_from_text(request.text) 

    created_todos = []

    with Session(engine) as session:
        for task in extracted.tasks:
            db_todo = Todo(
                title=task.title,
                completed=False,
                due_time=task.time,
            )   

            session.add(db_todo)   # 把新创建的 Todo 对象添加到 session 中，但还没有提交到数据库，所以它们还没有 id。
            created_todos.append(db_todo)   # 先把新创建的 Todo 对象添加到 session 中，但还没有提交到数据库，所以它们还没有 id。

        session.commit()    # 提交 session 中的所有更改到数据库，这时数据库会为每个新添加的 Todo 记录生成一个唯一的 id。

        for todo in created_todos:
            session.refresh(todo)   # 刷新每个 Todo 对象的状态，从数据库中获取它们最新的数据，包括生成的 id。

        logger.info("Created %d todos from AI extraction", len(created_todos))

        return {
            "extracted_tasks": extracted.tasks,
            "created_todos": created_todos,
        }
