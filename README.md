# FastAPI Todo API

My first FastAPI backend project.

This project is a simple Todo REST API built with FastAPI. It supports creating, reading, updating, and deleting todo items.

## Features

- FastAPI backend
- Todo CRUD API
- Request validation with Pydantic
- Response models
- HTTP status codes
- Swagger UI documentation
- Mock AI chat endpoint

## Tech Stack

- Python
- FastAPI
- Pydantic
- Uvicorn

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/about` | Project information |
| POST | `/echo` | Echo test endpoint |
| POST | `/chat` | Mock AI chat endpoint |
| GET | `/todos` | List all todos |
| POST | `/todos` | Create a todo |
| GET | `/todos/{todo_id}` | Get a todo by id |
| PATCH | `/todos/{todo_id}` | Update a todo |
| DELETE | `/todos/{todo_id}` | Delete a todo |

## Todo API Examples

### Create a Todo

Request:

```json
{
  "title": "Learn FastAPI"
}
```

Response:

```json
{
  "id": 3,
  "title": "Learn FastAPI",
  "completed": false
}
```

### List Todos

Request:

```text
GET /todos
```

Response:

```json
[
  {
    "id": 1,
    "title": "Learn FastAPI",
    "completed": false
  },
  {
    "id": 2,
    "title": "Build my first Todo API",
    "completed": false
  }
]
```

### Get a Todo by ID

Request:

```text
GET /todos/1
```

Response:

```json
{
  "id": 1,
  "title": "Learn FastAPI",
  "completed": false
}
```

### Update a Todo

Request:

```json
{
  "title": "Review FastAPI",
  "completed": true
}
```

Response:

```json
{
  "id": 1,
  "title": "Review FastAPI",
  "completed": true
}
```

### Delete a Todo

Request:

```text
DELETE /todos/1
```

Response:

```json
{
  "message": "Todo deleted"
}
```

## Current Version

`v0.2` - In-memory Todo CRUD API

## What I Learned

- How to create FastAPI routes
- How to use path parameters
- How to receive request body data
- How to validate input with Pydantic
- How to return JSON responses
- How to use HTTP status codes
- How to build a basic CRUD API

## Next Steps

- Add SQLite database
- Add tests with pytest
- Add Docker
- Add real AI chat API