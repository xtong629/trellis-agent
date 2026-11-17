import asyncio

import httpx
from experiments.pilot import PILOT_PICKS, HttpObserver, percentile, pilot_subset, summarize_http
from experiments.scenarios import taskset_a, taskset_b


def test_the_pilot_subset_is_stratified_and_deterministic() -> None:
    a, b = pilot_subset(taskset_a(), "a"), pilot_subset(taskset_b(), "b")
    assert [len(a["tasks"]), len(b["tasks"])] == [
        sum(PILOT_PICKS["a"].values()),
        sum(PILOT_PICKS["b"].values()),
    ]
    assert pilot_subset(taskset_a(), "a") == a
    families = {t["metadata"]["family"] for t in a["tasks"]}
    assert families == {"serial", "fanout", "flaky", "single"}
    assert {t["metadata"]["family"] for t in b["tasks"]} == set(PILOT_PICKS["b"])
    full_ids = {t["id"] for t in taskset_a()["tasks"]}
    assert {t["id"] for t in a["tasks"]} <= full_ids  # a subset, with unchanged task content
    assert a["name"] == "exp-a-planner-pilot"


def test_percentiles_use_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 90) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 50) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 90) == 9.0


def test_http_summary_counts_statuses_and_retries() -> None:
    log = [
        {"status": 200, "latency_s": 2.0, "in_flight": 1, "retry_after": None},
        {"status": 429, "latency_s": 0.2, "in_flight": 2, "retry_after": "3"},
        {"status": 200, "latency_s": 4.0, "in_flight": 2, "retry_after": None},
    ]
    s = summarize_http(log, model_calls=2)
    assert s["http_requests"] == 3 and s["status_counts"] == {"200": 2, "429": 1}
    assert s["rate_limited"] == 1 and s["extra_http_requests"] == 1  # 3 requests for 2 calls
    assert s["max_in_flight"] == 2 and s["retry_after_seen"] == ["3"]
    assert s["latency_s"]["max"] == 4.0 and s["latency_s"]["p50"] == 2.0


async def test_the_observer_records_status_and_latency_but_never_headers_or_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7", "X-Secret": "s3cret"}, text="no")

    observer = HttpObserver(max_requests=5)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks=observer.hooks()
    ) as client:
        await client.post("http://x.test/v1", headers={"Authorization": "Bearer KEY123"}, json={})
        await asyncio.sleep(0)
    (entry,) = observer.log
    assert entry["status"] == 429 and entry["retry_after"] == "7" and entry["latency_s"] >= 0
    assert "KEY123" not in str(observer.log) and "s3cret" not in str(observer.log)


async def test_the_observer_enforces_a_hard_request_cap() -> None:
    import pytest

    observer = HttpObserver(max_requests=1)
    transport = httpx.MockTransport(lambda r: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport, event_hooks=observer.hooks()) as client:
        await client.get("http://x.test/")
        with pytest.raises(RuntimeError, match="cap"):
            await client.get("http://x.test/")
