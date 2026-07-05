from __future__ import annotations

import json
import os
import sys
from uuid import uuid4

import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "support")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "support")
TIMEOUT_SECONDS = 120.0


def fail(message: str, *, response: httpx.Response | None = None) -> None:
    print(f"[FAIL] {message}")

    if response is not None:
        print(f"Status code: {response.status_code}")
        try:
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except Exception:
            print(response.text)

    sys.exit(1)


def require_status(
    response: httpx.Response,
    expected_status: int,
    step: str,
) -> None:
    if response.status_code != expected_status:
        fail(
            f"{step} expected HTTP {expected_status}, got {response.status_code}",
            response=response,
        )


def assert_contains_result(
    *,
    search_data: dict,
    document_id: str,
    filename: str,
    step: str,
) -> None:
    serialized = json.dumps(search_data, ensure_ascii=False)

    if document_id not in serialized and filename not in serialized:
        fail(
            f"{step} did not return the uploaded document. "
            f"Expected document_id={document_id} or filename={filename}. "
            f"Response preview: {serialized[:2000]}"
        )


def assert_not_contains_result(
    *,
    search_data: dict,
    document_id: str,
    filename: str,
    step: str,
) -> None:
    serialized = json.dumps(search_data, ensure_ascii=False)

    if document_id in serialized or filename in serialized:
        fail(
            f"{step} still returned the deleted document. "
            f"document_id={document_id}, filename={filename}. "
            f"Response preview: {serialized[:2000]}"
        )


def main() -> None:
    run_id = uuid4().hex[:8]
    filename = f"smoke_blue_whale_access_card_{run_id}.md"
    document_id: str | None = None
    deleted = False

    content = f"""# 蓝鲸门禁卡补办流程 {run_id}

如果员工的蓝鲸门禁卡丢失，需要先在行政系统登记遗失信息，
然后联系行政支持团队冻结旧卡，并提交补办申请。

补办申请必须包含员工姓名、部门、遗失时间、遗失地点和直属主管确认。

Smoke run id: {run_id}
""".encode("utf-8")

    query = "蓝鲸门禁卡丢失后如何补办？"

    print(f"API_BASE_URL={API_BASE_URL}")
    print(f"Smoke run id={run_id}")

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        try:
            print("1. Checking health...")
            response = client.get(f"{API_BASE_URL}/health")
            require_status(response, 200, "GET /health")
            print("   OK")

            response = client.post(
                f"{API_BASE_URL}/auth/token",
                json={"username": AUTH_USERNAME, "password": AUTH_PASSWORD},
            )
            require_status(response, 200, "POST /auth/token")
            headers = {
                "Authorization": f"Bearer {response.json()['access_token']}",
            }

            print("2. Uploading document...")
            response = client.post(
                f"{API_BASE_URL}/documents/upload",
                data={"category": "admin"},
                files={
                    "file": (
                        filename,
                        content,
                        "text/markdown",
                    )
                },
                headers=headers,
            )
            require_status(response, 201, "POST /documents/upload")
            uploaded = response.json()

            document_id = uploaded.get("id")
            if not document_id:
                fail("Upload response does not contain document id", response=response)

            if uploaded.get("status") != "uploaded":
                fail(
                    f"Expected uploaded status, got {uploaded.get('status')}",
                    response=response,
                )

            print(f"   OK document_id={document_id}")

            print("3. Indexing document...")
            response = client.post(
                f"{API_BASE_URL}/documents/{document_id}/index",
                headers=headers,
            )
            require_status(response, 200, "POST /documents/{document_id}/index")
            indexed = response.json()

            if indexed.get("status") != "indexed":
                fail(
                    f"Expected indexed status, got {indexed.get('status')}",
                    response=response,
                )

            if int(indexed.get("chunk_count", 0)) <= 0:
                fail("Expected chunk_count > 0 after indexing", response=response)

            print(f"   OK chunk_count={indexed.get('chunk_count')}")

            print("4. Searching uploaded document...")
            response = client.post(
                f"{API_BASE_URL}/rag/search",
                json={
                    "query": query,
                    "top_k": 5,
                    "category": "admin",
                },
                headers=headers,
            )
            require_status(response, 200, "POST /rag/search after index")
            search_data = response.json()

            assert_contains_result(
                search_data=search_data,
                document_id=document_id,
                filename=filename,
                step="POST /rag/search after index",
            )

            print("   OK uploaded document is retrievable")

            print("5. Deleting document...")
            response = client.delete(
                f"{API_BASE_URL}/documents/{document_id}",
                headers=headers,
            )
            require_status(response, 200, "DELETE /documents/{document_id}")
            deleted_data = response.json()

            if deleted_data.get("status") != "deleted":
                fail(
                    f"Expected deleted status, got {deleted_data.get('status')}",
                    response=response,
                )

            deleted_embeddings = int(deleted_data.get("deleted_embeddings", 0))
            if deleted_embeddings <= 0:
                fail(
                    f"Expected deleted_embeddings > 0, got {deleted_embeddings}",
                    response=response,
                )

            deleted = True
            print(f"   OK deleted_embeddings={deleted_embeddings}")

            print("6. Searching after delete...")
            response = client.post(
                f"{API_BASE_URL}/rag/search",
                json={
                    "query": query,
                    "top_k": 5,
                    "category": "admin",
                },
                headers=headers,
            )
            require_status(response, 200, "POST /rag/search after delete")
            search_after_delete = response.json()

            assert_not_contains_result(
                search_data=search_after_delete,
                document_id=document_id,
                filename=filename,
                step="POST /rag/search after delete",
            )

            print("   OK deleted document is no longer retrievable")

            print()
            print("Smoke test passed.")
            print(
                "Validated: health -> upload -> index -> search hit -> "
                "delete -> search miss"
            )

        finally:
            if document_id is not None and not deleted:
                print()
                print("Cleanup: deleting uploaded document after failed smoke run...")
                try:
                    client.delete(
                        f"{API_BASE_URL}/documents/{document_id}",
                        headers=headers,
                    )
                except Exception as exc:
                    print(f"Cleanup failed: {exc}")


if __name__ == "__main__":
    main()
