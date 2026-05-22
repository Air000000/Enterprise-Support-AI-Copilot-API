from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydanticField
from typing import Optional
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
from pathlib import Path

import os

from dotenv import load_dotenv
from openai import OpenAI

""" 
连接 SQLite 数据库
"""
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

sqlite_url = "sqlite:///data/todos.db"  # 在当前项目目录下创建或使用 todos.db 这个 SQLite 数据库文件

engine = create_engine(
    sqlite_url, # 连接哪个数据库
    echo=True,  # 把执行的 SQL 语句打印到终端
    connect_args={"check_same_thread": False},
)


# 定义一个 SQLModel 模型，表示数据库中的 "todo" 表格结构
class Todo(SQLModel, table=True):   #　table=True 表示这是一个数据库表格模型，表名默认为 "todo"
    id: Optional[int] = SQLField(default=None, primary_key=True)    # id可以一开始为空，它是主键。
    title: str
    completed: bool = False

# 根据定义的 SQLModel 表模型(class Todo)，在数据库里创建对应的数据表
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# 生命周期函数，在 FastAPI 启动时自动调用
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield   # yield 前面的代码：应用启动时执行。后面的代码：应用关闭时执行。


load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")

llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)


app = FastAPI(title="FastAPI Todo API", lifespan= lifespan) # 把 lifespan 传给 FastAPI 实例，应用启动时自动调用 create_db_and_tables() 来创建数据库表格。


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

class TodoUpdate(BaseModel):
    title: Optional[str] = PydanticField(default=None, min_length=1, max_length=100)
    completed: Optional[bool] = None

class TodoResponse(BaseModel):
    id: int
    title: str
    completed: bool

class EchoRequest(BaseModel):
    message: str = PydanticField(..., min_length=1, max_length=200)
    repeat: int = PydanticField(1, ge=1, le=5)


class ChatRequest(BaseModel):
    message: str = PydanticField(..., min_length=1, max_length=500)


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
    db_todo = Todo(
        title=todo.title,
        completed=False,
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

        session.add(todo)
        session.commit()
        session.refresh(todo)

        return todo


@app.delete("/todos/{todo_id}", status_code=200)
def delete_todo(todo_id: int):
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)

        if todo is None:
            raise HTTPException(status_code=404, detail="Todo not found")

        session.delete(todo)
        session.commit()

        return {"message": "Todo deleted"}
    
@app.post("/ai/chat")
def ai_chat(request: ChatRequest):
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DASHSCOPE_API_KEY is not configured",
        )

    try:
        completion = llm_client.chat.completions.create(
            model=DASHSCOPE_MODEL,
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
            "model": DASHSCOPE_MODEL,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM API error: {str(e)}",
        )