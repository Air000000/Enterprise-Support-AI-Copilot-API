"""Fixed fictional fixtures. Rubrics are local and excluded from wire inputs."""

FIXTURES = (
    ("single", "What is the cache timeout for fictional MapleCache release 14?", {
        1: "Fictional MapleCache release 14: the cache timeout is 30 seconds. This is the current complete timeout policy for release 14."
    }, "SUFFICIENT", "State 30 seconds for MapleCache14."),
    ("late_source", "How do I rotate a fictional JuniperKey 2.0 staging token?", {
        12: "Fictional JuniperKey 2.0 staging token rotation, complete procedure: open Settings, select Rotate token, and replace the staging token in the staging client. This procedure is for staging only; do not replace production tokens."
    }, "SUFFICIENT", "Retain Settings, Rotate token, client replacement and staging-only scope."),
    ("split_steps", "What are all steps to recover fictional BrookQueue 3.0 error B17 on a paused test queue?", {
        2: "Fictional BrookQueue 3.0 error B17 recovery for a paused test queue, part 1 of 2: clear the test queue lease. Complete recovery requires this step and the step in part 2.",
        9: "Fictional BrookQueue 3.0 error B17 recovery for a paused test queue, part 2 of 2: restart the test consumer after the lease has been cleared. There are no additional recovery steps. Do not use this procedure on production queues."
    }, "SUFFICIENT", "Clear lease then restart test consumer; retain test-only condition."),
    ("qualification", "How do I enable dry-run export for fictional CedarExport 4.0 Enterprise during maintenance?", {
        3: "Fictional CedarExport 4.0 Enterprise: during maintenance, set dry_run_export=true to enable dry-run export. Only Enterprise licenses support this setting. Do not perform production writes while maintenance is active."
    }, "SUFFICIENT", "Give dry_run_export=true, Enterprise/maintenance scope and no production writes."),
    ("scoped_alternative", "Give one supported way to download a fictional FernReport 5.0 test report.", {
        4: "Fictional FernReport 5.0: to download a test report, select Download in the test report panel. This is a complete supported method; it requires no CLI or administrator change.",
        11: "Fictional FernReport 4.0 only: export reports with the legacy fern-fetch CLI. This document does not cover release 5.0."
    }, "SUFFICIENT", "Use the release5 Download method; do not require the old CLI."),
    ("chinese", "虚构青松门禁的测试卡丢失后，需要如何补办？", {
        5: "虚构青松门禁测试卡补办的完整步骤：先在测试行政系统冻结旧测试卡，再到测试行政前台申请新测试卡。仅适用于测试环境，不得操作正式门禁卡。"
    }, "SUFFICIENT", "先冻结旧测试卡再申请新卡，保留仅测试环境限制。"),
    ("supported_no", "Does fictional WillowArchive release 14 support CSV export?", {
        8: "Fictional WillowArchive release 14 does not support CSV export. It supports JSON export only. This is the complete release14 export-format specification."
    }, "SUFFICIENT", "Answer no CSV, JSON only; a supported negative is not insufficient evidence."),
    ("similar_product", "What fixes DEMO-A17 in fictional PineGateway 2.0 test deployments?", {
        6: "Fictional PineGateway 2.0 test deployments: DEMO-A17 is corrected by patch PG20-T1. Apply PG20-T1 to the test gateway and restart that test gateway. These are the complete fix steps for this exact issue.",
        13: "Fictional PineConsole 2.0 test deployments: DEMO-A17 is corrected by patch PC20-T9. This is a different product; this patch must not be used for PineGateway."
    }, "SUFFICIENT", "Use PG20-T1 and test-gateway restart; do not borrow PineConsole's PC20-T9."),
    ("missing_outcome", "What is the maximum retention day count in fictional HazelStore 1.0?", {
        1: "Fictional HazelStore 1.0 supports creating and deleting test objects. This document does not specify any retention duration, day count or retention limit."
    }, "INSUFFICIENT", "No retention count may be invented."),
    ("version_mismatch", "Does fictional AcornExport version 2.0 support XML export?", {
        7: "Fictional AcornExport version 1.0 supports XML export. This documentation applies only to version 1.0 and establishes no behavior or compatibility for version 2.0."
    }, "INSUFFICIENT", "Do not infer version2 support from version1."),
    ("unknown_condition", "Can my fictional BirchTenant enable audit export? My license edition is unknown.", {
        10: "Fictional BirchTenant audit export is available only with an Enterprise license. It is unavailable with a Basic license. The tenant's license edition must be established before deciding whether it can enable this feature."
    }, "INSUFFICIENT", "Unknown tenant license leaves a material unresolved condition."),
    ("conflict", "What is the current fictional ElmCache 1.0 timeout for test workspaces?", {
        2: "Current authoritative fictional ElmCache 1.0 policy for all test workspaces: the timeout is 30 seconds. This policy has equal authority and the same effective date as the other current ElmCache1.0 timeout policy.",
        14: "Current authoritative fictional ElmCache 1.0 policy for all test workspaces: the timeout is 90 seconds. This policy has equal authority and the same effective date as the other current ElmCache1.0 timeout policy. No precedence rule is available."
    }, "INSUFFICIENT", "Conflicting equally applicable policies must not be resolved by guessing."),
)


def inputs():
    # ponytail: deterministic fixed context, no index, retrieval research or private files.
    return [{"id": name, "question": question, "sources": [
        {"source_id": f"Source {i}", "content": supplied.get(i,
            f"Fictional OtterLedger reference {i}: test ledger labels may use the color amber. "
            "This ledger document defines no behavior for other products or services.")}
        for i in range(1, 15)]} for name, question, supplied, _, _ in FIXTURES]
