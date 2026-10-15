"""Final-answer metrics and separate trace diagnostics.

A wrong trace with the right answer is correct. Format failures are wrong on
the main metric and are also reported on their own. Safety-limit failures are not converted into NO.
"""

from __future__ import annotations

from collections import defaultdict
from collections import Counter


def _recall(correct: int, support: int):
    if support == 0:
        return None
    return correct / support


def _pack(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    format_fail = sum(1 for row in rows if row["format_fail"])
    correct = sum(1 for row in rows if row["correct"] and not row["format_fail"])
    yes = [row for row in rows if row["label"] == 1]
    no = [row for row in rows if row["label"] == 0]
    yes_correct = sum(1 for row in yes if row["correct"] and not row["format_fail"])
    no_correct = sum(1 for row in no if row["correct"] and not row["format_fail"])
    yes_recall = _recall(yes_correct, len(yes))
    no_recall = _recall(no_correct, len(no))
    balanced = None
    if yes_recall is not None and no_recall is not None:
        balanced = 0.5 * (yes_recall + no_recall)
    return {
        "n": n,
        "n_yes": len(yes),
        "n_no": len(no),
        "n_correct": correct,
        "n_format_fail": format_fail,
        "answer_accuracy": correct / n,
        "balanced_accuracy": balanced,
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "format_fail_rate": format_fail / n,
    }


def aggregate(rows: list[dict]) -> dict:
    """``rows`` need correct, format_fail, label, and optional shape/distance/filler_tokens."""
    report = {"overall": _pack(rows)}
    for key in ("shape", "distance", "filler_tokens"):
        groups: dict = defaultdict(list)
        for row in rows:
            if key in row:
                groups[row[key]].append(row)
        report[f"by_{key}"] = {str(name): _pack(group) for name, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    return report


def prefix_length(got: list[int], reference: list[int]) -> int:
    n = 0
    for a, b in zip(got, reference):
        if a != b:
            break
        n += 1
    return n


def summarize_traces(rows: list[dict]) -> dict:
    """Independent of answer accuracy. Empty when the protocol has no DFS trace."""
    traced = [row for row in rows if "correct_prefix_tokens" in row]
    if not traced:
        return {"n": 0}
    n = len(traced)
    return {
        "n": n,
        "mean_correct_prefix_tokens": sum(row["correct_prefix_tokens"] for row in traced) / n,
        "generation_limit_rate": sum(1 for row in traced if row.get("generation_limit_reached")) / n,
        "search_complete_rate": sum(1 for row in traced if row.get("search_complete")) / n,
        "completed_records": sum(int(row.get("completed_records", 0)) for row in traced),
        "completed_records_match": sum(int(row.get("completed_records_match", 0)) for row in traced),
    }


class MetricAccumulator:
    """Keep counts rather than all predictions; batching never changes denominators."""

    def __init__(self):
        self.groups = {key: defaultdict(Counter) for key in ("overall", "shape", "distance", "filler_tokens")}
        self.trace = Counter()
        self.fields = Counter()

    def update(self, row: dict) -> None:
        for category in self.groups:
            key = "all" if category == "overall" else str(row[category])
            c = self.groups[category][key]
            good = bool(row["correct"] and not row["format_fail"])
            c["n"] += 1
            c["n_yes" if row["label"] == 1 else "n_no"] += 1
            c["n_correct"] += int(good)
            c["n_format_fail"] += int(row["format_fail"])
            c["yes_correct" if row["label"] == 1 else "no_correct"] += int(good)
        if "correct_prefix_tokens" in row:
            self.trace["n"] += 1
            for field in ("correct_prefix_tokens", "generation_limit_reached", "search_complete", "completed_records",
                          "completed_records_match", "malformed_completed_records"):
                self.trace[field] += int(row.get(field, 0))
            self.fields.update(row.get("field_correct", {}))

    @staticmethod
    def _finish(c) -> dict:
        n = c["n"]
        if not n:
            return {"n": 0}
        yes = _recall(c["yes_correct"], c["n_yes"])
        no = _recall(c["no_correct"], c["n_no"])
        return {**{k: c[k] for k in ("n", "n_yes", "n_no", "n_correct", "n_format_fail")},
                "answer_accuracy": c["n_correct"] / n,
                "balanced_accuracy": None if yes is None or no is None else (yes + no) / 2,
                "yes_recall": yes, "no_recall": no, "format_fail_rate": c["n_format_fail"] / n}

    def result(self) -> dict:
        result = {"overall": self._finish(self.groups["overall"]["all"])}
        for category in ("shape", "distance", "filler_tokens"):
            result[f"by_{category}"] = {k: self._finish(v) for k, v in sorted(self.groups[category].items())}
        n = self.trace["n"]
        trace = {"n": n}
        if n:
            denom = self.trace["completed_records"]
            trace.update(mean_correct_prefix_tokens=self.trace["correct_prefix_tokens"] / n,
                         generation_limit_rate=self.trace["generation_limit_reached"] / n,
                         search_complete_rate=self.trace["search_complete"] / n,
                         completed_records=denom,
                         completed_records_match=self.trace["completed_records_match"],
                         malformed_completed_records=self.trace["malformed_completed_records"],
                         field_correct=dict(self.fields),
                         field_accuracy={k: (v / denom if denom else None) for k, v in self.fields.items()})
        result["trace"] = trace
        return result
