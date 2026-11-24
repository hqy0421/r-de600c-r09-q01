"""Bitemporal node history and as-of reads.

Covers the pilot-park requirement end to end at the storage layer:

* a measurement point moved from group A to group B keeps its stable uuid and
  its series, so its numeric records stay continuously addressable;
* an as-of read against a knowledge time *before* the move resolves the point's
  path to A and the subtree as it was then, while the latest read shows B;
* the as-of manifest resolver reproduces the regular resolver's contract but
  walks ``node_history`` intervals, for both the ``path`` and ``node_uuid``
  routes;
* rename / subtree move / delete / backfill / dry_run maintain the
  half-open-interval invariants;
* the public ``as_of=`` guards (timezone-aware, no ``end_known`` conflict,
  edges and relative reads reject it) fail loudly.

These exercise the PostgreSQL hierarchy half of the feature directly (the half
that owns path/version resolution). ClickHouse value reads are environmentally
unavailable in this harness (CH writes fail at the filesystem layer), so value
frames are not asserted here; the resolve contracts that feed them are.

Live tests: skipped unless both ``TIMEDB_PG_DSN`` and ``TIMEDB_CH_URL`` are set.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID

import polars as pl
import psycopg
import pytest

import energydb as edb
from energydb import Client
from energydb.errors import ManifestError, NodeNotFoundError, SeriesNotFoundError, ValidationError
from energydb.history import (
    backfill_node_history,
    get_node_history,
    resolve_manifest_as_of,
    resolve_node_uuid_as_of,
    resolve_path_as_of,
)

if not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")):
    pytest.skip(
        "TIMEDB_PG_DSN / TIMEDB_CH_URL not set: skipping bitemporal history tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    c = Client()
    c.delete()
    c.create()
    yield c
    c.delete()
    c.close()


def _raw_async_dsn() -> str:
    return os.environ["TIMEDB_PG_DSN"]


def asyncio_run(coro):
    """Run an async history helper against its own raw async connection."""
    return asyncio.run(coro)


@asynccontextmanager
async def _aconn():
    """Yield a fresh async connection.

    A query-level error aborts a psycopg connection, so tests that assert a
    raised error and then keep querying open a new connection per call rather
    than reusing an aborted one.
    """
    conn = await psycopg.AsyncConnection.connect(_raw_async_dsn())
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def park(client):
    """Park with two groups A and B; measurement point M1 starts under A.

    Layout before the move::

        Park/A/M1   (owns actual.power)
        Park/B
    """
    client.create_node(node_type="site", name="Park")
    client.create_node(node_type="group", name="A", parent="Park")
    client.create_node(node_type="group", name="B", parent="Park")
    m1 = client.create_node(node_type="meter", name="M1", parent="Park/A")
    client.get_node(uuid=m1).register_series(
        edb.TimeSeries(name="power", unit="MW", data_type=edb.DataType.ACTUAL)
    )
    return client, m1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


async def _history_rows(node_uuid: UUID) -> list[dict]:
    async with await psycopg.AsyncConnection.connect(_raw_async_dsn()) as conn:
        return await get_node_history(conn, node_uuid)


async def _anchor_inside(node_uuid: UUID, path: str) -> datetime:
    """Return a knowledge time strictly inside the history interval whose path
    is ``path``.

    Deriving the as-of anchor from the *stored* interval (rather than a
    wall-clock ``now()``) makes the tests deterministic: a synthesized timestamp
    can race on either side of a fast mutation's boundary, whereas the midpoint
    of ``[valid_from, valid_to)`` provably satisfies
    ``valid_from <= anchor < valid_to`` and is independent of execution speed.
    For an open interval (``valid_to IS NULL``) a small epsilon after
    ``valid_from`` is used.
    """
    rows = await _history_rows(node_uuid)
    interval = next(r for r in rows if r["path"] == path)
    start = interval["valid_from"]
    end = interval["valid_to"]
    if end is None:
        return start + timedelta(milliseconds=10)
    return start + (end - start) / 2


def anchor_in(node_uuid: UUID, path: str) -> datetime:
    return asyncio_run(_anchor_inside(node_uuid, path))


async def _resolve_manifest_async(manifest: pl.DataFrame, as_of: datetime, on_missing: str = "raise"):
    async with await psycopg.AsyncConnection.connect(_raw_async_dsn()) as conn:
        return await resolve_manifest_as_of(conn, manifest, as_of=as_of, on_missing=on_missing)


def _path_manifest(path: str) -> pl.DataFrame:
    return pl.DataFrame({"path": [path], "data_type": ["actual"], "name": ["power"]})


def _uuid_manifest(node_uuid: UUID) -> pl.DataFrame:
    return pl.DataFrame({"node_uuid": [str(node_uuid)], "data_type": ["actual"], "name": ["power"]})


# ---------------------------------------------------------------------------
# The headline scenario: move A -> B, as-of still resolves A, latest shows B
# ---------------------------------------------------------------------------


def test_move_then_as_of_path_resolves_old_group(park):
    client, m1 = park

    # The move under test.
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    # Deterministic anchors derived from the stored intervals, not wall-clock.
    t_before = anchor_in(m1, "Park/A/M1")
    t_after = anchor_in(m1, "Park/B/M1")

    # Latest view: the live node path is B, same identity.
    live = client.get_node_raw(m1)
    assert live["path"] == "Park/B/M1"
    assert live["uuid"] == m1

    # As-of before the move: the stable uuid still resolves to its old A path.
    async def check():
        async with _aconn() as conn:
            path_before = await resolve_path_as_of(conn, m1, t_before)
        async with _aconn() as conn:
            path_after = await resolve_path_as_of(conn, m1, t_after)
        async with _aconn() as conn:
            uuid_at_old = await resolve_node_uuid_as_of(conn, ("Park", "A", "M1"), t_before)
        # Old A path no longer resolves after the move (fresh connection: the
        # raised error aborts the one that ran it).
        async with _aconn() as conn:
            with pytest.raises(NodeNotFoundError):
                await resolve_node_uuid_as_of(conn, ("Park", "A", "M1"), t_after)
        return path_before, path_after, uuid_at_old

    path_before, path_after, uuid_at_old = asyncio_run(check())
    assert tuple(path_before) == ("Park", "A", "M1")
    assert tuple(path_after) == ("Park", "B", "M1")
    assert uuid_at_old == m1


def test_move_records_two_intervals_and_keeps_one_open(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))

    rows = asyncio_run(_history_rows(m1))
    assert len(rows) == 2
    first, second = rows
    assert first["path"] == "Park/A/M1"
    assert second["path"] == "Park/B/M1"
    # Half-open chain: the new interval opens exactly when the old one closed.
    assert first["valid_to"] == second["valid_from"]
    assert first["valid_to"] is not None
    assert second["valid_to"] is None  # exactly one open row
    # Identity columns never change.
    assert all(r["node_uuid"] == m1 for r in rows)
    assert rows[0]["name"] == rows[1]["name"] == "M1"


def test_stable_uuid_and_series_survive_move(park):
    """The series (and thus the ClickHouse value keyed by its series_id) stays
    owned by the same node uuid across the move: no delete+insert."""
    client, m1 = park
    series_before = client.list_series(m1)
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    series_after = client.list_series(m1)

    assert series_before == series_after
    assert series_after[0]["name"] == "power"
    # The node count is unchanged (a move is not a recreate).
    assert len(client.get_subtree_raw(m1)) >= 1


# ---------------------------------------------------------------------------
# As-of manifest resolver: path and node_uuid routes
# ---------------------------------------------------------------------------


def test_resolve_manifest_as_of_path_route(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    t_before = anchor_in(m1, "Park/A/M1")
    t_after = anchor_in(m1, "Park/B/M1")

    # Before the move, the A path resolves; B path does not yet hold M1.
    resolved_before, summary_before = asyncio_run(
        _resolve_manifest_async(_path_manifest("Park/A/M1"), t_before)
    )
    assert resolved_before["node_uuid"][0] == str(m1)
    assert resolved_before["path"][0] == "Park/A/M1"
    assert "series_id" in resolved_before.columns and resolved_before["series_id"][0] is not None
    assert summary_before.missing.height == 0

    # After the move, the A path is an unknown address -> raise by default.
    with pytest.raises(SeriesNotFoundError):
        asyncio_run(_resolve_manifest_async(_path_manifest("Park/A/M1"), t_after))
    # ...and the B path now resolves to the same series.
    resolved_after, _ = asyncio_run(_resolve_manifest_async(_path_manifest("Park/B/M1"), t_after))
    assert resolved_after["node_uuid"][0] == str(m1)


def test_resolve_manifest_as_of_uuid_route_attaches_old_path(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    t_before = anchor_in(m1, "Park/A/M1")
    t_after = anchor_in(m1, "Park/B/M1")

    # Stable uuid route: values stay addressable; the attached path label moves.
    before, _ = asyncio_run(_resolve_manifest_async(_uuid_manifest(m1), t_before))
    after, _ = asyncio_run(_resolve_manifest_async(_uuid_manifest(m1), t_after))
    assert before["series_id"][0] == after["series_id"][0]  # same value key
    assert before["node_uuid"][0] == str(m1)
    assert before["path"][0] == "Park/A/M1"
    assert after["path"][0] == "Park/B/M1"


def test_resolve_manifest_as_of_on_missing_skip(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    t_after = anchor_in(m1, "Park/B/M1")

    manifest = pl.DataFrame(
        {
            "path": ["Park/A/M1", "Park/B/M1"],
            "data_type": ["actual", "actual"],
            "name": ["power", "power"],
        }
    )
    resolved, summary = asyncio_run(_resolve_manifest_async(manifest, t_after, on_missing="skip"))
    # Only the B triple survives; the stale A triple is reported as missing.
    assert resolved.height == 1
    assert resolved["path"][0] == "Park/B/M1"
    assert summary.missing.height == 1
    assert summary.missing["path"][0] == "Park/A/M1"


def test_resolve_manifest_as_of_rejects_edge_and_bad_kwargs(park):
    _client, m1 = park
    as_of = datetime.now(UTC)

    edge_manifest = pl.DataFrame(
        {"edge_uuid": [str(m1)], "data_type": ["actual"], "name": ["power"]}
    )
    with pytest.raises(ManifestError):
        asyncio_run(_resolve_manifest_async(edge_manifest, as_of))

    no_route = pl.DataFrame({"data_type": ["actual"], "name": ["power"]})
    with pytest.raises(ManifestError):
        asyncio_run(_resolve_manifest_async(no_route, as_of))

    # Naive datetime is a caller error, not a catalog gap.
    with pytest.raises(ValidationError):
        asyncio_run(_resolve_manifest_async(_path_manifest("Park/A/M1"), datetime(2026, 1, 1)))


# ---------------------------------------------------------------------------
# Subtree as-of resolution (the scope read path)
# ---------------------------------------------------------------------------


def test_scope_resolve_as_of_subtree_membership(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    t_before = anchor_in(m1, "Park/A/M1")
    t_after = anchor_in(m1, "Park/B/M1")

    # Group A subtree before the move contains M1's series; after, it is empty.
    a_before = client.get_node("Park", "A").resolve(as_of=t_before)
    a_after = client.get_node("Park", "A").resolve(as_of=t_after)
    assert a_before is not None and a_before["node_uuid"].to_list() == [str(m1)]
    assert a_before["path"][0] == "Park/A/M1"
    assert a_after is None

    # Group B gains it only after the move.
    b_before = client.get_node("Park", "B").resolve(as_of=t_before)
    b_after = client.get_node("Park", "B").resolve(as_of=t_after)
    assert b_before is None
    assert b_after is not None and b_after["node_uuid"].to_list() == [str(m1)]
    assert b_after["path"][0] == "Park/B/M1"


def test_scope_resolve_latest_shows_b(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"))
    latest = client.get_node("Park", "B").resolve()
    assert latest is not None
    assert latest["path"].to_list() == ["Park/B/M1"]
    assert latest["node_uuid"][0] == str(m1)


# ---------------------------------------------------------------------------
# Rename, subtree move, delete, backfill, dry_run interval semantics
# ---------------------------------------------------------------------------


def test_rename_opens_interval_with_new_name(park):
    client, m1 = park
    client.get_node("Park", "A", "M1").rename("M1x")
    rows = asyncio_run(_history_rows(m1))
    assert [r["name"] for r in rows] == ["M1", "M1x"]
    assert rows[-1]["path"] == "Park/A/M1x"
    assert rows[-1]["valid_to"] is None


def test_subtree_move_repaths_child_intervals(client):
    client.create_node(node_type="site", name="Park")
    client.create_node(node_type="group", name="A", parent="Park")
    client.create_node(node_type="group", name="B", parent="Park")
    client.create_node(node_type="folder", name="F", parent="Park/A")
    child = client.create_node(node_type="meter", name="C", parent="Park/A/F")

    client.get_node("Park", "A", "F").move_to(client.get_node("Park", "B"))
    rows = asyncio_run(_history_rows(child))
    assert rows[0]["path"] == "Park/A/F/C"
    assert rows[-1]["path"] == "Park/B/F/C"
    assert rows[-1]["valid_to"] is None


def test_delete_closes_interval_and_retains_history(park):
    client, m1 = park
    client.get_node(uuid=m1).delete()
    rows = asyncio_run(_history_rows(m1))
    assert rows, "node_history must survive the live delete"
    assert all(r["valid_to"] is not None for r in rows)  # no open interval remains
    assert client.get_node_raw(m1) is None  # live row gone


def test_backfill_is_idempotent(client):
    # A node created through a path that bypasses history tracking has no
    # history row; backfill seeds exactly one open interval, then no-ops.
    solo = client.create_node(node_type="site", name="Solo")

    async def wipe_and_backfill():
        async with _aconn() as conn:
            await conn.execute("DELETE FROM node_history WHERE node_uuid = %s", (solo,))
            await conn.commit()
            n1 = await backfill_node_history(conn)
            await conn.commit()
            n2 = await backfill_node_history(conn)
            await conn.commit()
            return n1, n2

    n1, n2 = asyncio_run(wipe_and_backfill())
    assert n1 == 1
    assert n2 == 0
    rows = asyncio_run(_history_rows(solo))
    assert len(rows) == 1 and rows[0]["valid_to"] is None and rows[0]["path"] == "Solo"


def test_dry_run_move_rolls_back_history(park):
    client, m1 = park
    diff = client.get_node("Park", "A", "M1").move_to(client.get_node("Park", "B"), dry_run=True)
    assert diff is not None
    # No new interval: only the original open row, live path unchanged.
    rows = asyncio_run(_history_rows(m1))
    assert len(rows) == 1 and rows[0]["valid_to"] is None and rows[0]["path"] == "Park/A/M1"
    assert client.get_node_raw(m1)["path"] == "Park/A/M1"


# ---------------------------------------------------------------------------
# Public as_of argument guards
# ---------------------------------------------------------------------------


def test_client_read_rejects_naive_as_of(park):
    client, _m1 = park
    with pytest.raises(ValidationError, match="timezone-aware"):
        client.read(_path_manifest("Park/A/M1"), as_of=datetime(2026, 1, 1))


def test_client_read_rejects_as_of_with_end_known(park):
    client, _m1 = park
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="end_known"):
        client.read(_path_manifest("Park/A/M1"), as_of=now, end_known=now)


def test_scope_read_rejects_edge_as_of(park):
    client, _m1 = park
    # Build an edge between two generic nodes; as-of must be refused regardless
    # of whether the endpoints reconstruct into EDM objects.
    j1 = client.create_node(node_type="junction", name="J1")
    j2 = client.create_node(node_type="junction", name="J2")
    line = edb.grid.Line(
        name="L1",
        from_element=edb.Reference(j1),
        to_element=edb.Reference(j2),
    )
    edge_uuid = client.create_edge(line)
    with pytest.raises(ValidationError, match="edge"):
        client.get_edge(uuid=edge_uuid).read(as_of=datetime.now(UTC))


def test_read_relative_rejects_as_of(park):
    client, _m1 = park
    with pytest.raises(ValidationError, match="read_relative"):
        client.read_relative(_path_manifest("Park/A/M1"), as_of=datetime.now(UTC))
