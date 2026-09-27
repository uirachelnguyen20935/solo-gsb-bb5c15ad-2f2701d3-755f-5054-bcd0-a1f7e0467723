"""Cross-service direct-call edges for one trace.

An edge is a *direct* parent/child call whose two ends belong to different,
non-empty services (``parent_span_id`` matches a span that actually arrived).
For every qualifying edge the **call count** is aggregated by
``(parent_service, child_service)``. The edge's consumed time is measured per
parent span and per target service: the child's half-open ``[start, end)``
interval is intersected with the parent's interval, the intersections of all
children calling the same target service are unioned (so overlapping siblings
count only once), and the union lengths are summed across parent spans.

Spans whose parent never arrived contribute nothing; same-service parent/child
pairs and self references are not cross-service edges; empty (zero-length)
intersections count as a call but contribute no time. Pure functions only —
no I/O, no sampling policy.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .servicetime import _union_length


def service_edges(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate direct cross-service call edges over one trace's spans.

    ``rows`` are mappings with at least ``span_id``, ``parent_span_id``,
    ``service``, ``start_ns`` and ``end_ns`` keys (``sqlite3.Row`` works).
    Returns a list of ``{"source_service", "target_service", "call_count",
    "duration_ns"}`` sorted by duration ns descending, then source and target
    service name ascending.
    """
    spans = [
        (
            r["span_id"],
            r["parent_span_id"],
            r["service"],
            int(r["start_ns"]),
            int(r["end_ns"]),
        )
        for r in rows
    ]
    by_id: dict[str, tuple[str | None, int, int]] = {
        sid: (service, start_ns, end_ns)
        for sid, _pid, service, start_ns, end_ns in spans
    }

    # (source, target) -> call count
    counts: dict[tuple[str, str], int] = {}
    # (parent span id, target service) -> clipped child intersections, plus
    # the parent's service carried alongside for the final summation.
    per_parent: dict[tuple[str, str], dict[str, Any]] = {}

    for span_id, parent_id, service, start_ns, end_ns in spans:
        # Only relationships whose parent span already arrived count; a
        # missing parent, a self reference, or an empty service on either end
        # cannot be a cross-service edge.
        if parent_id is None or parent_id == span_id or parent_id not in by_id:
            continue
        parent_service, p_start, p_end = by_id[parent_id]
        if not service or not parent_service or service == parent_service:
            continue

        key = (parent_service, service)
        counts[key] = counts.get(key, 0) + 1

        lo = max(start_ns, p_start)
        hi = min(end_ns, p_end)
        if hi > lo:  # half-open intersection; zero length contributes no time
            bucket = per_parent.setdefault(
                (parent_id, service),
                {"source": parent_service, "overlaps": []},
            )
            bucket["overlaps"].append((lo, hi))

    duration: dict[tuple[str, str], int] = {}
    for (_parent_id, target), bucket in per_parent.items():
        key = (bucket["source"], target)
        duration[key] = duration.get(key, 0) + _union_length(bucket["overlaps"])

    result = [
        {
            "source_service": source,
            "target_service": target,
            "call_count": counts[(source, target)],
            "duration_ns": duration.get((source, target), 0),
        }
        for (source, target) in counts
    ]
    result.sort(key=lambda e: (
        -e["duration_ns"], e["source_service"], e["target_service"],
    ))
    return result
