from typing import Literal
from fastapi import APIRouter, Depends, Query

from schemas.agent_ops import (
    AgentOpsMetricsSummaryResponse,
    AgentRunResponse,
    AgentRunTraceResponse,
    ApprovalDecisionRequest,
    ApprovalRequestResponse,
    ApprovalRequestUpdate,
    RetrievalFailureMetricResponse,
    RetrievalLogResponse,
    RetrievalMetricsSummaryResponse,
    RetrievalNoContextQueryMetricResponse,
    RetrievalSourceMetricResponse,
    ToolCallResponse,
)
from services.agent_ops_service import (
    get_agent_ops_metrics_summary as get_agent_ops_metrics_summary_service,
    get_agent_run as get_agent_run_service,
    get_agent_run_trace as get_agent_run_trace_service,
    get_retrieval_failure_metrics as get_retrieval_failure_metrics_service,
    get_retrieval_metrics_summary as get_retrieval_metrics_summary_service,
    get_retrieval_no_context_query_metrics as get_retrieval_no_context_query_metrics_service,
    get_retrieval_source_metrics as get_retrieval_source_metrics_service,
    list_agent_runs as list_agent_runs_service,
    list_approval_requests as list_approval_requests_service,
    list_approval_requests_by_run as list_approval_requests_by_run_service,
    list_retrieval_logs as list_retrieval_logs_service,
    list_tool_calls as list_tool_calls_service,
    list_tool_calls_by_run as list_tool_calls_by_run_service,
    update_approval_request as update_approval_request_service,
)
from auth import CurrentUser, require_roles

router = APIRouter(prefix="/agent-ops", tags=["agent-ops"])


AgentRunStatusQuery = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
]

ToolCallStatusQuery = Literal[
    "pending",
    "success",
    "failed",
]

ApprovalRequestStatusQuery = Literal[
    "pending",
    "approved",
    "rejected",
    "cancelled",
]

ApprovalTypeQuery = Literal[
    "ticket_creation",
]

RetrievalEndpointQuery = Literal[
    "search",
    "ask",
]

RetrievalStatusQuery = Literal[
    "ok",
    "no_context",
    "refused_low_relevance",
    "failed",
]


@router.get("/runs", response_model=list[AgentRunResponse])
def list_agent_runs(
    status: AgentRunStatusQuery | None = None,
    agent_name: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[AgentRunResponse]:
    agent_runs = list_agent_runs_service(
        tenant_id=user.tenant_id,
        status=status,
        agent_name=agent_name,
        limit=limit,
        offset=offset,
    )

    return [
        AgentRunResponse.model_validate(agent_run)
        for agent_run in agent_runs
    ]


@router.get(
    "/metrics/retrieval/no-context-queries",
    response_model=list[RetrievalNoContextQueryMetricResponse],
)
def get_retrieval_no_context_query_metrics(
    endpoint: RetrievalEndpointQuery | None = None,
    category: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[RetrievalNoContextQueryMetricResponse]:
    return get_retrieval_no_context_query_metrics_service(
        tenant_id=user.tenant_id,
        endpoint=endpoint,
        category=category,
        limit=limit,
    )


@router.get(
    "/metrics/retrieval/failures",
    response_model=list[RetrievalFailureMetricResponse],
)
def get_retrieval_failure_metrics(
    endpoint: RetrievalEndpointQuery | None = None,
    category: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[RetrievalFailureMetricResponse]:
    return get_retrieval_failure_metrics_service(
        tenant_id=user.tenant_id,
        endpoint=endpoint,
        category=category,
        limit=limit,
    )


@router.get("/runs/{agent_run_id}", response_model=AgentRunResponse)
def get_agent_run(
    agent_run_id: int,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> AgentRunResponse:
    agent_run = get_agent_run_service(
        agent_run_id=agent_run_id,
        tenant_id=user.tenant_id,
    )

    return AgentRunResponse.model_validate(agent_run)


@router.get(
    "/runs/{agent_run_id}/trace",
    response_model=AgentRunTraceResponse,
)
def get_agent_run_trace(
    agent_run_id: int,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> AgentRunTraceResponse:
    agent_run, tool_calls, approval_requests = get_agent_run_trace_service(
        agent_run_id=agent_run_id,
        tenant_id=user.tenant_id,
    )

    return AgentRunTraceResponse(
        agent_run=AgentRunResponse.model_validate(agent_run),
        tool_calls=[
            ToolCallResponse.model_validate(tool_call)
            for tool_call in tool_calls
        ],
        approval_requests=[
            ApprovalRequestResponse.model_validate(approval_request)
            for approval_request in approval_requests
        ],
    )


@router.get("/tool-calls", response_model=list[ToolCallResponse])
def list_tool_calls(
    agent_run_id: int | None = None,
    status: ToolCallStatusQuery | None = None,
    tool_name: str | None = None,
    error_type: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[ToolCallResponse]:
    tool_calls = list_tool_calls_service(
        tenant_id=user.tenant_id,
        agent_run_id=agent_run_id,
        status=status,
        tool_name=tool_name,
        error_type=error_type,
        limit=limit,
        offset=offset,
    )

    return [
        ToolCallResponse.model_validate(tool_call)
        for tool_call in tool_calls
    ]


@router.get(
    "/runs/{agent_run_id}/tool-calls",
    response_model=list[ToolCallResponse],
)
def list_tool_calls_by_run(
    agent_run_id: int,
    status: ToolCallStatusQuery | None = None,
    tool_name: str | None = None,
    error_type: str | None = None,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[ToolCallResponse]:
    tool_calls = list_tool_calls_by_run_service(
        agent_run_id=agent_run_id,
        tenant_id=user.tenant_id,
        status=status,
        tool_name=tool_name,
        error_type=error_type,
    )

    return [
        ToolCallResponse.model_validate(tool_call)
        for tool_call in tool_calls
    ]


@router.get(
    "/approval-requests",
    response_model=list[ApprovalRequestResponse],
)
def list_approval_requests(
    agent_run_id: int | None = None,
    status: ApprovalRequestStatusQuery | None = None,
    approval_type: ApprovalTypeQuery | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[ApprovalRequestResponse]:
    approval_requests = list_approval_requests_service(
        tenant_id=user.tenant_id,
        agent_run_id=agent_run_id,
        status=status,
        approval_type=approval_type,
        limit=limit,
        offset=offset,
    )

    return [
        ApprovalRequestResponse.model_validate(approval_request)
        for approval_request in approval_requests
    ]


@router.get(
    "/runs/{agent_run_id}/approval-requests",
    response_model=list[ApprovalRequestResponse],
)
def list_approval_requests_by_run(
    agent_run_id: int,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[ApprovalRequestResponse]:
    approval_requests = list_approval_requests_by_run_service(
        agent_run_id=agent_run_id,
        tenant_id=user.tenant_id,
    )

    return [
        ApprovalRequestResponse.model_validate(approval_request)
        for approval_request in approval_requests
    ]


@router.get(
    "/retrieval-logs",
    response_model=list[RetrievalLogResponse],
)
def list_retrieval_logs(
    endpoint: RetrievalEndpointQuery | None = None,
    retrieval_status: RetrievalStatusQuery | None = None,
    category: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[RetrievalLogResponse]:
    retrieval_logs = list_retrieval_logs_service(
        tenant_id=user.tenant_id,
        endpoint=endpoint,
        retrieval_status=retrieval_status,
        category=category,
        limit=limit,
        offset=offset,
    )

    return [
        RetrievalLogResponse.model_validate(retrieval_log)
        for retrieval_log in retrieval_logs
    ]


@router.get(
    "/metrics/summary",
    response_model=AgentOpsMetricsSummaryResponse,
)
def get_agent_ops_metrics_summary(
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> AgentOpsMetricsSummaryResponse:
    return get_agent_ops_metrics_summary_service(
        tenant_id=user.tenant_id,
    )


@router.get(
    "/metrics/retrieval",
    response_model=RetrievalMetricsSummaryResponse,
)
def get_retrieval_metrics_summary(
    endpoint: RetrievalEndpointQuery | None = None,
    category: str | None = None,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> RetrievalMetricsSummaryResponse:
    return get_retrieval_metrics_summary_service(
        tenant_id=user.tenant_id,
        endpoint=endpoint,
        category=category,
    )


@router.get(
    "/metrics/retrieval/sources",
    response_model=list[RetrievalSourceMetricResponse],
)
def get_retrieval_source_metrics(
    endpoint: RetrievalEndpointQuery | None = None,
    category: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> list[RetrievalSourceMetricResponse]:
    return get_retrieval_source_metrics_service(
        tenant_id=user.tenant_id,
        endpoint=endpoint,
        category=category,
        limit=limit,
    )


@router.post(
    "/approval-requests/{approval_request_id}/reject",
    response_model=ApprovalRequestResponse,
)
def reject_approval_request(
    approval_request_id: int,
    request: ApprovalDecisionRequest,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> ApprovalRequestResponse:
    approval_request = update_approval_request_service(
        approval_request_id=approval_request_id,
        tenant_id=user.tenant_id,
        approval_request_update=ApprovalRequestUpdate(
            status="rejected",
            approved_by=user.user_id,
            decision_reason=request.reason,
        ),
    )

    return ApprovalRequestResponse.model_validate(approval_request)


@router.post(
    "/approval-requests/{approval_request_id}/cancel",
    response_model=ApprovalRequestResponse,
)
def cancel_approval_request(
    approval_request_id: int,
    request: ApprovalDecisionRequest,
    user: CurrentUser = Depends(require_roles("support", "admin")),
) -> ApprovalRequestResponse:
    approval_request = update_approval_request_service(
        approval_request_id=approval_request_id,
        tenant_id=user.tenant_id,
        approval_request_update=ApprovalRequestUpdate(
            status="cancelled",
            approved_by=user.user_id,
            decision_reason=request.reason,
        ),
    )

    return ApprovalRequestResponse.model_validate(approval_request)
