"""Bounded, no-motion tests of model decisions, not robot task success.

Three labeled synthetic contract fixtures and two unlabelled recorded scenes.
Gold expectations are kept outside model inputs; no response-driven retries.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .agent import PlannerDeferred, PlanningRequest, agent_program_json_schema
from .bailian import BailianClient, BailianError, BailianStructuredPlanner, MODEL
from .hammer_chain import HAMMER_CHAIN, make_hammer_phase_manifests
from .planning_snapshot import TASK, PlanningSnapshot, load_hammer_snapshot


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    source_kind: str
    snapshot: PlanningSnapshot
    expected_actions: tuple[str, ...] | None

    def __post_init__(self):
        if not self.case_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.case_id):
            raise ValueError("invalid decision test case identifier")
        if self.source_kind not in ("synthetic_contract_fixture", "recorded_rgbd_snapshot"):
            raise ValueError("unknown decision case source kind")
        if self.source_kind == "recorded_rgbd_snapshot" and self.expected_actions is not None:
            raise ValueError("recorded ambiguous scenes must not be assigned synthetic gold actions")
        if self.source_kind == "synthetic_contract_fixture" and (
            not self.expected_actions or any(action not in ("program", "reobserve", "stop") for action in self.expected_actions)
        ):
            raise ValueError("synthetic fixtures need explicit, valid evaluation expectations")

    def to_dict(self):
        return {"case_id": self.case_id, "source_kind": self.source_kind,
                "expected_actions": self.expected_actions,
                "evaluation_scope": ("stipulated interface behavior only; not vision competence"
                                     if self.expected_actions is not None else
                                     "static contract and descriptive decision only; no semantic gold label")}


def synthetic_cases():
    manifests = make_hammer_phase_manifests(task_id=TASK, expected_action_dim=12)
    catalog = tuple(m.to_dict() for m in sorted(manifests, key=lambda m: m.name))
    # Stipulated interface values, not modified measurements from a real scene.
    parameters = {"target_hammer_pose7": [-.25, -.375, 1.0625, .5, .5, .5, .5],
                  "target_nail_pose7": [.0625, .125, 1.125, .5, .5, .5, .5],
                  "target_pose_source": "synthetic_contract_fixture/no_robot"}
    base = {
        "source_kind": "synthetic_contract_fixture", "historical_snapshot": False,
        "physical_effect_verified": False, "required_skill_chain": HAMMER_CHAIN.to_dict(),
        "fixture_description": "Numerical protocol fixture. No image is expected. No real object or robot exists.",
        "fixture_stipulations": {"successful_source_candidate_available": True,
                                 "reset_orientation_condition_met": True,
                                 "pose_pair_provided_without_ambiguity": True},
        "evidence_state": {"schema_version": 1,
                           "current_pose_observation": {"status": "accepted_estimate", "source_event_id": "fixture_new"},
                           "unresolved_perception_feedback_count": 0,
                           "physical_effect_status": "unverified"},
        "scene_estimate": {"hammer_pose7": parameters["target_hammer_pose7"],
                           "nail_pose7": parameters["target_nail_pose7"],
                           "backend": parameters["target_pose_source"]},
        "planning_gate": {"program_allowed": True, "allowed_actions": ["program", "reobserve", "stop"],
                          "reason": "stipulated inputs permit constructing a conditional plan, not physical execution"},
    }
    historical = {"stage": "perception", "scope": "earlier_perception_attempt",
                  "code": "ambiguous", "reason": "an earlier fixture view lacked a pose pair",
                  "source_event_id": "fixture_old", "superseded_by_event_id": "fixture_new",
                  "temporal_status": "superseded_for_reset_pose_source", "current_blocker": False,
                  "physical_problem_resolved": False}
    cases = []
    for case_id, expected in (("fixture_program", "program"), ("fixture_observe", "reobserve"),
                              ("fixture_stop", "stop")):
        summary = deepcopy(base)
        locked = {m.name: deepcopy(parameters) for m in manifests}
        feedback = (deepcopy(historical),)
        objective = (
            "在明确标注的合成接口样例中，依据给定的当前有效信息提出下一步。"
            "有完整无歧义的数值输入时组装条件性技能计划；缺少输入且还有观测机会时先观察；"
            "观测机会耗尽且禁止动作时停止。不要执行，不要宣称实际视觉或抓取成功。"
        )
        if expected != "program":
            summary["scene_estimate"] = None
            summary["fixture_stipulations"]["pose_pair_provided_without_ambiguity"] = False
            summary["evidence_state"]["current_pose_observation"] = {"status": "missing_or_ambiguous"}
            summary["evidence_state"]["unresolved_perception_feedback_count"] = 1
            locked = {}
            feedback = ({"stage": "perception", "reason": "current pose pair is unavailable",
                         "temporal_status": "current_unresolved", "current_blocker": True},)
            summary["observation_opportunities_remaining"] = 1 if expected == "reobserve" else 0
            summary["planning_gate"] = {
                "program_allowed": False,
                "allowed_actions": ["reobserve", "stop"] if expected == "reobserve" else ["stop"],
                "reason": ("current evidence unavailable; one read-only view opportunity remains"
                           if expected == "reobserve" else "no current pose pair and observation budget is exhausted"),
            }
        request = PlanningRequest(task_id=TASK, objective=objective, observation_summary=summary,
                                  skill_catalog=catalog, locked_parameters=locked, prior_feedback=feedback,
                                  constraints=("synthetic interface test, not real visual evidence",
                                               "copy all supplied locked parameters exactly",
                                               "no actuation, code execution, VLA or simulator queries"))
        cases.append(DecisionCase(case_id, "synthetic_contract_fixture",
                                  PlanningSnapshot(request, {"fixture_version": 1}), (expected,)))
    return tuple(cases)


def build_decision_cases(trace_path):
    return (*synthetic_cases(),
            DecisionCase("recorded_reset", "recorded_rgbd_snapshot", load_hammer_snapshot(trace_path), None),
            DecisionCase("recorded_blocked", "recorded_rgbd_snapshot", load_hammer_snapshot(trace_path, mode="blocked"), None))


def _write_json(path, value, client):
    path.write_text(json.dumps(client.redact(value), ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _aggregate(report, client):
    report["calls_attempted"] = client.calls_attempted
    report["api_audit"] = client.audit
    report["reported_tokens"] = sum(row.get("usage", {}).get("total_tokens", 0) for row in client.audit)
    report["usage_complete"] = all("total_tokens" in row.get("usage", {}) for row in client.audit)
    report["synthetic_expectation_passes"] = sum(row.get("expectation_result") == "pass" for row in report["cases"])
    report["synthetic_case_count"] = sum(row["source_kind"] == "synthetic_contract_fixture" for row in report["cases"])
    report["recorded_contract_valid_count"] = sum(row.get("contract_valid") is True and
                                                row["source_kind"] == "recorded_rgbd_snapshot" for row in report["cases"])


def run_decision_suite(cases, *, output: Path, client: BailianClient, call_api=False):
    """One shared client budget across all cases, including failed requests."""
    if not cases or len(cases) > 5 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("suite needs one to five unique cases")
    if client.calls_attempted or client.audit or client.budget.maximum_calls > 5:
        raise ValueError("suite requires a fresh client with at most five calls")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "summary.json"
    if report_path.exists() or any((output / case.case_id).exists() for case in cases):
        raise ValueError("refusing to overwrite an existing decision suite")
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live_api_no_motion" if call_api else "offline_preflight", "status": "preparing",
        "model": MODEL, "maximum_api_calls": client.budget.maximum_calls,
        "maximum_completion_tokens_per_call": client.budget.maximum_completion_tokens,
        "robot_actions": 0, "new_observations": 0, "execution_enabled": False,
        "billing_verified": False, "real_task_success_rate": None,
        "semantic_accuracy_on_recorded_scenes": None,
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("agent.py", "bailian.py", "planning_snapshot.py", "decision_evaluation.py")},
        "cases": [{**case.to_dict(), "status": "not_run", "expectation_result": "not_run"} for case in cases],
    }
    prepared = []
    # Prepare all inputs before spending any API calls. Gold metadata is local only.
    for case in cases:
        folder = output / case.case_id
        folder.mkdir()
        planner = BailianStructuredPlanner(client=client)
        agent = case.snapshot.make_agent(planner)
        schema = agent_program_json_schema(task_id=case.snapshot.request.task_id, manifests=agent._manifests(),
                                          limits=agent.limits, chain_contract=agent.chain_contract)
        payload = planner.build_payload(case.snapshot.request, output_schema=schema)
        for name, value in (("case.json", case.to_dict()), ("request.json", case.snapshot.request.to_dict()),
                            ("api_request.json", payload), ("provenance.json", case.snapshot.provenance)):
            _write_json(folder / name, value, client)
        for i, image in enumerate(case.snapshot.request.images):
            (folder / f"observation_{i}.png").write_bytes(image.png)
        prepared.append((folder, planner, agent))
    report["status"] = "running" if call_api else "offline_preflight"
    _aggregate(report, client)
    _write_json(report_path, report, client)
    transport_failed = False
    interrupted = False
    for case, row, (folder, planner, agent) in zip(cases, report["cases"], prepared):
        if not call_api:
            row["status"] = "not_called"
        elif interrupted or transport_failed:
            row["status"] = "not_run_after_error"
        elif client.calls_attempted >= client.budget.maximum_calls:
            row["status"] = "not_run_budget"
        else:
            before = client.calls_attempted
            try:
                decision = agent.plan_only(request=case.snapshot.request, context=case.snapshot.context())
                row.update(status="plan_validated", action="program", contract_valid=True, decision=decision.to_dict())
            except PlannerDeferred as deferred:
                row.update(status="deferred", action=deferred.action, contract_valid=True, advice=deferred.to_dict())
            except (BailianError, ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
                row.update(status="error", contract_valid=False, error_type=type(exc).__name__,
                           error=str(client.redact(str(exc)))[:2048])
                # JSON/contract failures are results, never retried. Transport/auth
                # failures stop the rest of the suite to avoid repeated charges.
                transport_failed = (client.calls_attempted == before or
                                    client.audit[-1].get("status") != "response_received")
            except KeyboardInterrupt:
                interrupted = True
                row.update(status="interrupted", contract_valid=False)
            row["api_calls"] = client.calls_attempted - before
            row["model_response"] = planner.last_response
            if "action" not in row and planner.last_advice is not None:
                row["action"] = planner.last_advice["action"]
            row["expectation_result"] = (
                "not_scored" if case.expected_actions is None else
                "pass" if row.get("contract_valid") and row.get("action") in case.expected_actions else "fail")
        _aggregate(report, client)
        _write_json(folder / "result.json", row, client)
        _write_json(report_path, report, client)
        print(f"{case.case_id}: {row['status']}; action={row.get('action', 'none')}; "
              f"expectation={row['expectation_result']}", flush=True)
    if call_api:
        report["status"] = ("interrupted" if interrupted else "partial" if any(
            row["status"].startswith("not_run") for row in report["cases"]) else
            "completed_with_failures" if any(row["status"] == "error" or row["expectation_result"] == "fail"
                                             for row in report["cases"]) else "completed")
    _write_json(report_path, report, client)
    return report
