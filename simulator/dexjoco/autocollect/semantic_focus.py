"""One-candidate workload probe with exactly the same images and split prompt."""
import json

from jsonschema import Draft202012Validator

from .agent import _canonical_json, _sha256
from .bailian import BailianError
from .program import _reject_json_constant, _unique_object
from .semantic_ablation import SemanticAblationReviewer

FOCUS_SCHEDULE = ((1, "V2C12"), (1, "V2C03"), (1, "V2C01"),
                  (2, "V2C01"), (2, "V2C03"), (2, "V2C12"))


class SemanticFocusReviewer(SemanticAblationReviewer):
    def build_payload(self, package, candidate_id):
        payload = super().build_payload(package, "blind_split")
        context = json.loads(payload["messages"][1]["content"][0]["text"])
        evidence = context["visual_evidence"]
        matches = [c for c in evidence["candidates"] if c["candidate_id"] == candidate_id]
        if len(matches) != 1:
            raise ValueError("focus must reference one existing candidate")
        evidence["candidates"] = matches
        for view in evidence["views"]:
            view["candidate_ids"] = [i for i in view["candidate_ids"] if i == candidate_id]
        schema = context["output_schema"]
        schema["properties"]["review_id"]["const"] = _sha256({"protocol": "semantic_focus_v1", "context": evidence})
        rows = schema["properties"]["candidate_reviews"]
        rows.update(minItems=1, maxItems=1)
        rows["items"]["properties"]["candidate_id"]["enum"] = [candidate_id]
        payload["messages"][1]["content"][0]["text"] = _canonical_json(context).decode()
        return payload

    def parse_response(self, content, package, candidate_id):
        if not isinstance(content, str) or len(content.encode()) > self.client.budget.maximum_response_bytes:
            raise BailianError("单候选响应类型无效或超过预算。")
        try:
            value = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
            payload = self.build_payload(package, candidate_id)
            schema = json.loads(payload["messages"][1]["content"][0]["text"])["output_schema"]
            if next(Draft202012Validator(schema).iter_errors(value), None):
                raise ValueError
            row = value["candidate_reviews"][0]
            if (row["mask_extent"] in ("target_only", "target_and_other") and row["requested_part_presence"] != "present"
                    or row["mask_extent"] == "other_only" and row["requested_part_presence"] != "absent"):
                raise ValueError
        except (ValueError, KeyError, TypeError, RecursionError):
            raise BailianError("单候选回答的格式、引用或字段一致性无效；不修复、不重试。") from None
        return {"status": "response_validated", "validation_scope": "format, references and field consistency only",
                "suggestions": value, "semantic_identity_verified": False, "actuation_allowed": False,
                "training_eligible": False, "reobservation_executed": False}
