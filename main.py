from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="FastAPI Todo API")


todos = [
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


class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)

class TodoUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=100)
    completed: Optional[bool] = None

class TodoResponse(BaseModel):
    id: int
    title: str
    completed: bool

class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=200)
    repeat: int = Field(1, ge=1, le=5)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


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
    return todos

@app.post("/todos", status_code=201, response_model=TodoResponse)    # 创建成功时，返回 201 Created
def create_todo(todo: TodoCreate):
    global next_id

    new_todo = {
        "id": next_id,
        "title": todo.title,
        "completed": False,
    }

    todos.append(new_todo)
    next_id += 1

    return new_todo

@app.get("/todos/{todo_id}", response_model=TodoResponse)
def get_todo(todo_id: int):
    for todo in todos:
        if todo["id"] == todo_id:
            return todo

    raise HTTPException(status_code=404, detail="Todo not found")

'''
TodoUpdate 是表格模板
update 是根据用户提交的 JSON 填好的那张表
update.completed 是这张表里 completed 那一栏的值
'''
@app.patch("/todos/{todo_id}", response_model=TodoResponse)
def update_todo(todo_id: int, update: TodoUpdate):
    for todo in todos:
        if todo["id"] == todo_id:
            if update.title is not None:
                todo["title"] = update.title

            if update.completed is not None:
                todo["completed"] = update.completed

            return todo

    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/todos/{todo_id}", status_code = 200) # 括号里是需要接收的输入 {}表示是一个变量
def delete_todo(todo_id: int):
    for todo in todos:
        if todo["id"] == todo_id:
            todos.remove(todo)
            return {"message": "Todo deleted"}

    raise HTTPException(status_code=404, detail="Todo not found")