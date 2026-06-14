from fastapi import APIRouter

from schemas.agent_ops import (
    AgentRunResponse,
    ApprovalRequestResponse,
    ToolCallResponse,
)
from services.agent_ops_service import (
    get_agent_run as get_agent_run_service,
    list_agent_runs as list_agent_runs_service,
    list_approval_requests_by_run as list_approval_requests_by_run_service,
    list_tool_calls_by_run as list_tool_calls_by_run_service,
)


router = APIRouter(prefix="/agent-ops", tags=["agent-ops"])

MOCK_TENANT_ID = "tenant_demo"


@router.get("/runs", response_model=list[AgentRunResponse])
def list_agent_runs(
    status: str | None = None,
    agent_name: str | None = None,
) -> list[AgentRunResponse]:
    agent_runs = list_agent_runs_service(
        tenant_id=MOCK_TENANT_ID,
        status=status,
        agent_name=agent_name,
    )

    return [
        AgentRunResponse.model_validate(agent_run)
        for agent_run in agent_runs
    ]


@router.get("/runs/{agent_run_id}", response_model=AgentRunResponse)
def get_agent_run(agent_run_id: int) -> AgentRunResponse:
    agent_run = get_agent_run_service(
        agent_run_id=agent_run_id,
        tenant_id=MOCK_TENANT_ID,
    )

    return AgentRunResponse.model_validate(agent_run)


@router.get(
    "/runs/{agent_run_id}/tool-calls",
    response_model=list[ToolCallResponse],
)
def list_tool_calls_by_run(
    agent_run_id: int,
) -> list[ToolCallResponse]:
    tool_calls = list_tool_calls_by_run_service(
        agent_run_id=agent_run_id,
        tenant_id=MOCK_TENANT_ID,
    )

    return [
        ToolCallResponse.model_validate(tool_call)
        for tool_call in tool_calls
    ]


@router.get(
    "/runs/{agent_run_id}/approval-requests",
    response_model=list[ApprovalRequestResponse],
)
def list_approval_requests_by_run(
    agent_run_id: int,
) -> list[ApprovalRequestResponse]:
    approval_requests = list_approval_requests_by_run_service(
        agent_run_id=agent_run_id,
        tenant_id=MOCK_TENANT_ID,
    )

    return [
        ApprovalRequestResponse.model_validate(approval_request)
        for approval_request in approval_requests
    ]