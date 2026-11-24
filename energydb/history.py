"""Bitemporal node-history: system-period intervals and as-of hierarchy reads.

The live :class:`~energydb.models.Node` row holds only the *current* structural
state. This module owns the append-only :class:`~energydb.models.NodeHistory`
table that keeps every prior state, so the asset path a node had at any
knowledge time is recoverable after a move or rename.

Time model

* Each structural state of a node is one row valid over the half-open interval
  ``[valid_from, valid_to)`` of *knowledge time* (when the structure was known
  to be true), mirroring TimeDB's knowledge-time axis for forecast versions.
* Exactly one row per node has ``valid_to IS NULL``: its current state.
* All interval boundaries in one logical mutation share a single server-side
  transaction timestamp (``clock_timestamp()`` captured once), so a subtree
  move opens/ closes every affected node at the same instant and the as-of
  view never sees a half-applied subtree.

The node ``uuid`` never changes across intervals; only ``parent_uuid`` /
``path`` (and ``name`` on a rename) do. Time-series values in ClickHouse are
keyed by the owner uuid, so they stay continuously addressable while the as-of
path moves with the hierarchy.

Pure structural changes are recorded (move, rename, create, delete). A
``data`` JSONB / ``node_type`` edit does not change asset path and so does not
open an interval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import polars as pl

from energydb.errors import ManifestError, NodeNotFoundError, ValidationError
from energydb.models import SQL_SCHEMA_PREFIX as P
from energydb.paths import (
    ResolveSummary,
    _check_on_missing,
    _missing_frame,
    _overlapping_ids,
    _resolvable_keys,
    _series_not_found_error,
)

# An as-of row's shape as returned by the as-of resolvers below. ``valid_to``
# is None for the row that is still current at the queried time.
NodeHistoryRow = dict[str, Any]


async def transaction_now(conn) -> datetime:
    """Capture one server-side transaction timestamp for a whole mutation.

    ``clock_timestamp()`` (not ``now()``) is deliberate: inside a transaction
    ``now()``/``transaction_timestamp()`` returns the transaction's start time,
    which would equal an earlier statement's boundary when several mutations
    share a connection. ``clock_timestamp()`` advances per statement, so each
    mutation gets a distinct, monotonic boundary while every row touched by
    *this* call shares the single value captured here.
    """
    row = await (await conn.execute("SELECT clock_timestamp()")).fetchone()
    return row[0]


async def open_node_interval(
    conn,
    *,
    node_uuid: UUID,
    node_type: str,
    name: str,
    parent_uuid: UUID | None,
    path: str,
    valid_from: datetime,
) -> None:
    """Record a node's initial structural state (node creation).

    Idempotent on the assumption that creates are create-only: inserts the open
    interval with ``valid_to = NULL``. A node never created through the history
    path can be backfilled with :func:`backfill_node_history`.
    """
    await conn.execute(
        f"""
        INSERT INTO {P}node_history
            (node_uuid, valid_from, node_type, name, parent_uuid, path, valid_to)
        VALUES (%s, %s, %s, %s, %s, %s, NULL)
        """,
        (node_uuid, valid_from, node_type, name, parent_uuid, path),
    )


async def close_node_interval(conn, *, node_uuid: UUID, valid_to: datetime) -> None:
    """Close a node's currently-open interval at ``valid_to`` (node deletion)."""
    await conn.execute(
        f"UPDATE {P}node_history SET valid_to = %s WHERE node_uuid = %s AND valid_to IS NULL",
        (valid_to, node_uuid),
    )


async def replace_subtree_intervals(
    conn,
    *,
    old_path: str,
    new_path: str,
    renamed_node_uuid: UUID | None = None,
    new_name: str | None = None,
    valid_from: datetime,
) -> int:
    """Close current intervals and open new ones for a re-pathed subtree.

    Handles both :meth:`NodeScope.move_to` (``renamed_node_uuid=None``) and
    :meth:`NodeScope.rename` (the subtree root's ``name`` changes too). Every
    node whose current path is ``old_path`` or beneath it gets, in one
    transaction:

    1. its open history row closed at ``valid_from``;
    2. a new open row carrying the rewritten ``path`` (prefix swap), the same
       ``parent_uuid`` / ``node_type``, and the new ``name`` only on the root
       when this is a rename.

    Set-based: one multi-row INSERT ... SELECT driven by the rows being closed,
    so a subtree of any size costs a fixed number of statements. Returns the
    number of nodes whose interval was replaced.
    """
    from energydb.paths import _like_escape

    # Close the open intervals of every node in the moved/renamed subtree.
    await conn.execute(
        f"""
        UPDATE {P}node_history
        SET valid_to = %s
        WHERE valid_to IS NULL
          AND (path = %s OR path LIKE %s || '/%%' ESCAPE '\\')
        """,
        (valid_from, old_path, _like_escape(old_path)),
    )

    # Open a fresh interval per affected node, copying the just-closed state and
    # swapping the path prefix. The root's name changes only on an explicit
    # rename (renamed_node_uuid IS NOT NULL); descendant names, parent_uuid and
    # node_type all carry over. COALESCE keeps the CASE branches uniformly text.
    if renamed_node_uuid is not None:
        name_sql = "CASE WHEN h.node_uuid = %s THEN %s::text ELSE h.name END"
        name_params: list[Any] = [renamed_node_uuid, new_name]
    else:
        name_sql = "h.name"
        name_params = []

    await conn.execute(
        f"""
        INSERT INTO {P}node_history
            (node_uuid, valid_from, node_type, name, parent_uuid, path, valid_to)
        SELECT h.node_uuid, %s, h.node_type,
               {name_sql},
               h.parent_uuid,
               %s || substring(h.path FROM length(%s) + 1),
               NULL
        FROM {P}node_history h
        WHERE h.valid_to = %s
          AND (h.path = %s OR h.path LIKE %s || '/%%' ESCAPE '\\')
        """,
        (
            valid_from,
            *name_params,
            new_path,
            old_path,
            valid_from,
            old_path,
            _like_escape(old_path),
        ),
    )

    row = await (
        await conn.execute(
            f"""
            SELECT count(*) FROM {P}node_history
            WHERE valid_from = %s
              AND (path = %s OR path LIKE %s || '/%%' ESCAPE '\\')
            """,
            (valid_from, new_path, _like_escape(new_path)),
        )
    ).fetchone()
    return int(row[0])


async def backfill_node_history(conn) -> int:
    """Backfill open intervals for live nodes missing history rows.

    Migration/repair helper: a node created before history tracking (or by a
    path that bypassed the history write) gets one open interval seeded from
    its current live row. Rows that already have an open interval are left
    untouched. Returns the number of backfilled nodes.
    """
    row = await (
        await conn.execute(
            f"""
            INSERT INTO {P}node_history
                (node_uuid, valid_from, node_type, name, parent_uuid, path, valid_to)
            SELECT n.uuid, n.created_at, n.node_type, n.name, n.parent_uuid, n.path, NULL
            FROM {P}node n
            WHERE NOT EXISTS (
                SELECT 1 FROM {P}node_history h WHERE h.node_uuid = n.uuid
            )
            RETURNING node_uuid
            """
        )
    ).fetchall()
    return len(row)


def _check_as_of(as_of: datetime | None) -> None:
    """Validate an as-of knowledge-time argument (must be timezone-aware)."""
    if as_of is None:
        return
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise ValidationError("as_of must be a timezone-aware datetime or None")


async def resolve_node_uuid_as_of(
    conn,
    path: tuple[str, ...],
    as_of: datetime,
) -> UUID:
    """Resolve a materialized path to the node uuid that occupied it at ``as_of``.

    As-of counterpart of :func:`energydb.paths.resolve_node_uuid`. A path can
    belong to different nodes over time (a node moved out, another in), so this
    returns the node whose history interval covers ``as_of`` and whose path then
    matched. Raises :class:`~energydb.errors.NodeNotFoundError` if no node held
    that path at that knowledge time.
    """
    _check_as_of(as_of)
    joined = "/".join(path)
    row = await (
        await conn.execute(
            f"""
            SELECT node_uuid FROM {P}node_history
            WHERE path = %s
              AND valid_from <= %s
              AND (valid_to IS NULL OR valid_to > %s)
            ORDER BY valid_from DESC
            LIMIT 1
            """,
            (joined, as_of, as_of),
        )
    ).fetchone()
    if row is None:
        raise NodeNotFoundError(
            f"Node not found at as_of={as_of.isoformat()}: {joined}",
            path=joined,
        )
    return row[0]


async def resolve_path_as_of(conn, node_uuid: UUID, as_of: datetime) -> tuple[str, ...]:
    """Return the path tuple a node had at ``as_of``.

    As-of counterpart of :func:`energydb.paths.resolve_path`. Raises
    :class:`~energydb.errors.NodeNotFoundError` if the node has no interval
    covering that knowledge time (e.g. queried before it was created, or after
    it was deleted).
    """
    _check_as_of(as_of)
    row = await (
        await conn.execute(
            f"""
            SELECT path FROM {P}node_history
            WHERE node_uuid = %s
              AND valid_from <= %s
              AND (valid_to IS NULL OR valid_to > %s)
            ORDER BY valid_from DESC
            LIMIT 1
            """,
            (node_uuid, as_of, as_of),
        )
    ).fetchone()
    if row is None:
        raise NodeNotFoundError(
            f"Node not found at as_of={as_of.isoformat()}: uuid={node_uuid}",
            uuid=node_uuid,
        )
    return tuple(row[0].split("/"))


async def _as_of_meta_by_path(
    conn,
    manifest: pl.DataFrame,
    *,
    as_of: datetime,
    on_missing: str,
) -> tuple[pl.DataFrame, ResolveSummary]:
    """Path-routed as-of resolve, mirroring ``_resolve_manifest_by_path``.

    The live resolve joins ``node`` (current paths only); this joins
    ``node_history`` intervals covering ``as_of`` instead. Series ownership is
    by stable node uuid, so the same ``series`` rows attach regardless of where
    the node lives today. The manifest's literal ``path`` is the address that
    was valid at ``as_of`` — a path that only exists now will not resolve.
    """
    if manifest["path"].dtype == pl.List(pl.Utf8):
        raise ManifestError(
            "Manifest 'path' column must be Utf8 joined with '/'. "
            "Got List(Utf8): pass 'a/b/c' instead of ['a','b','c']."
        )
    if manifest["path"].dtype != pl.Utf8:
        raise ManifestError(f"Manifest 'path' column must be Utf8 joined with '/'; got {manifest['path'].dtype}.")
    if manifest.height - manifest["path"].null_count() == 0:
        raise ManifestError("No path values to resolve in manifest.")
    if manifest["path"].null_count() > 0:
        null_row = manifest.filter(pl.col("path").is_null()).row(0, named=True)
        raise ManifestError(
            f"Series not registered for path=None, data_type={null_row['data_type']!r}, name={null_row['name']!r}."
        )

    manifest = manifest.with_columns(manifest.select(["path", "data_type", "name"]).hash_rows().alias("_triple_k"))
    triples_df = manifest.select(["path", "data_type", "name", "_triple_k"]).unique(subset=["_triple_k"])
    miss_keys = triples_df["_triple_k"].to_list()
    miss_triples: list[tuple[str, str, str]] = list(
        zip(triples_df["path"].to_list(), triples_df["data_type"].to_list(), triples_df["name"].to_list(), strict=True)
    )

    unique_paths = list({t[0] for t in miss_triples})
    rows = await (
        await conn.execute(
            f"""
            SELECT h.path, s.data_type, s.name, s.series_id, s.canonical_unit,
                   s.timeseries_type, s.retention, h.node_uuid::text
            FROM {P}node_history h
            JOIN {P}series s ON s.node_uuid = h.node_uuid
            WHERE h.valid_from <= %s
              AND (h.valid_to IS NULL OR h.valid_to > %s)
              AND h.path = ANY(%s)
            """,
            (as_of, as_of, unique_paths),
        )
    ).fetchall()
    # Meta tuple layout matches _resolve_manifest_by_path(attach_path=True):
    # (series_id, canonical_unit, timeseries_type, retention, node_uuid).
    triple_to_meta: dict[tuple[str, str, str], tuple] = {}
    for path, dt, name, sid, unit, ts_type, retention, node_uuid in rows:
        triple_to_meta[(path, dt, name)] = (sid, unit, ts_type, retention, node_uuid)

    unresolved = [t for t in miss_triples if t not in triple_to_meta]
    if unresolved and on_missing == "raise":
        raise _series_not_found_error(route="path", cols=("path", "data_type", "name"), unresolved=unresolved)

    hash_to_meta: dict[int, tuple] = {
        hk: triple_to_meta[t] for hk, t in zip(miss_keys, miss_triples, strict=True) if t in triple_to_meta
    }
    if unresolved:
        manifest = manifest.join(_resolvable_keys(hash_to_meta), on="_triple_k", how="semi")

    metas = list(hash_to_meta.values())
    lookup_df = pl.DataFrame(
        {
            "_triple_k": list(hash_to_meta.keys()),
            "series_id": [m[0] for m in metas],
            "retention": [m[3] for m in metas],
            "canonical_unit": [m[1] for m in metas],
            # Re-attach the as-of path and its node uuid from the resolved meta
            # (the manifest path already equals h.path, but node_uuid is new).
            "node_uuid": [m[4] for m in metas],
        },
        schema={
            "_triple_k": pl.UInt64,
            "series_id": pl.Int64,
            "retention": pl.Utf8,
            "canonical_unit": pl.Utf8,
            "node_uuid": pl.Utf8,
        },
    )
    overlap = [c for c in lookup_df.columns if c != "_triple_k" and c in manifest.columns]
    if overlap:
        manifest = manifest.drop(overlap)
    resolved = manifest.join(lookup_df, on="_triple_k", how="left").drop("_triple_k")
    summary = ResolveSummary(
        overlapping_series_ids=_overlapping_ids(hash_to_meta),
        missing=_missing_frame(("path", "data_type", "name"), unresolved),
    )
    return resolved, summary


async def _as_of_meta_by_owner(
    conn,
    manifest: pl.DataFrame,
    *,
    as_of: datetime,
    on_missing: str,
) -> tuple[pl.DataFrame, ResolveSummary]:
    """node_uuid-routed as-of resolve, mirroring ``_resolve_manifest_by_owner``.

    The stable uuid is the series owner and never moves, so the ``series``
    catalog is read exactly as in the live path. What changes is the ``path``
    label: it comes from the history interval covering ``as_of`` rather than the
    live ``node`` row. A uuid with no interval covering ``as_of`` (the node did
    not exist yet, or had already been deleted) resolves to nothing, just like a
    path that did not exist then.
    """
    if manifest["node_uuid"].dtype != pl.Utf8:
        values = [str(v) if v is not None else None for v in manifest["node_uuid"].to_list()]
        manifest = manifest.with_columns(pl.Series("node_uuid", values, dtype=pl.Utf8))

    if manifest.height - manifest["node_uuid"].null_count() == 0:
        raise ManifestError("No node_uuid values to resolve in manifest.")
    if manifest["node_uuid"].null_count() > 0:
        null_row = manifest.filter(pl.col("node_uuid").is_null()).row(0, named=True)
        raise ManifestError(
            f"Series not registered for node_uuid=None, data_type={null_row['data_type']!r}, name={null_row['name']!r}."
        )

    manifest = manifest.with_columns(manifest.select(["node_uuid", "data_type", "name"]).hash_rows().alias("_triple_k"))
    triples_df = manifest.select(["node_uuid", "data_type", "name", "_triple_k"]).unique(subset=["_triple_k"])
    miss_keys = triples_df["_triple_k"].to_list()
    miss_triples: list[tuple[str, str, str]] = list(
        zip(triples_df["node_uuid"].to_list(), triples_df["data_type"].to_list(), triples_df["name"].to_list(), strict=True)
    )

    unique_owners = list({t[0] for t in miss_triples})
    # One query: as-of path from the covering history interval, joined to the
    # live series catalog by the stable owner uuid. An INNER join on history
    # drops series whose owner had no interval at as_of.
    rows = await (
        await conn.execute(
            f"""
            SELECT s.node_uuid::text, s.data_type, s.name, s.series_id,
                   s.canonical_unit, s.timeseries_type, s.retention, h.path
            FROM {P}node_history h
            JOIN {P}series s ON s.node_uuid = h.node_uuid
            WHERE h.valid_from <= %s
              AND (h.valid_to IS NULL OR h.valid_to > %s)
              AND s.node_uuid = ANY(%s::uuid[])
            """,
            (as_of, as_of, unique_owners),
        )
    ).fetchall()
    # Meta tuple: (series_id, canonical_unit, timeseries_type, retention, path).
    triple_to_meta: dict[tuple[str, str, str], tuple] = {}
    for owner, dt, name, sid, unit, ts_type, retention, path in rows:
        triple_to_meta[(owner, dt, name)] = (sid, unit, ts_type, retention, path)

    unresolved = [t for t in miss_triples if t not in triple_to_meta]
    if unresolved and on_missing == "raise":
        raise _series_not_found_error(
            route="node_uuid", cols=("node_uuid", "data_type", "name"), unresolved=unresolved
        )

    hash_to_meta: dict[int, tuple] = {
        hk: triple_to_meta[t] for hk, t in zip(miss_keys, miss_triples, strict=True) if t in triple_to_meta
    }
    if unresolved:
        manifest = manifest.join(_resolvable_keys(hash_to_meta), on="_triple_k", how="semi")

    metas = list(hash_to_meta.values())
    lookup_df = pl.DataFrame(
        {
            "_triple_k": list(hash_to_meta.keys()),
            "series_id": [m[0] for m in metas],
            "retention": [m[3] for m in metas],
            "canonical_unit": [m[1] for m in metas],
            # The path the stable node held at as_of (overwrites any supplied one).
            "path": [m[4] for m in metas],
        },
        schema={
            "_triple_k": pl.UInt64,
            "series_id": pl.Int64,
            "retention": pl.Utf8,
            "canonical_unit": pl.Utf8,
            "path": pl.Utf8,
        },
    )
    overlap = [c for c in lookup_df.columns if c != "_triple_k" and c in manifest.columns]
    if overlap:
        manifest = manifest.drop(overlap)
    resolved = manifest.join(lookup_df, on="_triple_k", how="left").drop("_triple_k")
    summary = ResolveSummary(
        overlapping_series_ids=_overlapping_ids(hash_to_meta),
        missing=_missing_frame(("node_uuid", "data_type", "name"), unresolved),
    )
    return resolved, summary


async def resolve_manifest_as_of(
    conn,
    manifest: pl.DataFrame,
    *,
    as_of: datetime,
    on_missing: str = "raise",
) -> tuple[pl.DataFrame, ResolveSummary]:
    """Resolve a node-routed manifest against the hierarchy known at ``as_of``.

    Bitemporal counterpart of :func:`energydb.paths.resolve_manifest` for the
    two node routes (``path`` and ``node_uuid``). Edge-routed and edge-triple
    manifests are rejected: edge endpoint history is not modeled, so an as-of
    edge read has no structural basis and must fail loudly rather than silently
    read current topology.

    For a ``path`` manifest the literal path is matched against
    ``node_history`` intervals covering ``as_of`` (a path that only exists today
    will not resolve). For a ``node_uuid`` manifest the stable uuid is mapped to
    the path it held at ``as_of``, so values stay continuously addressable
    across a move while the returned ``path`` label reflects where the node
    lived then.

    Returns the manifest with ``series_id`` / ``retention`` /
    ``canonical_unit`` / ``node_uuid`` / ``path`` attached plus a
    :class:`ResolveSummary` — the same contract as the regular resolver run with
    ``attach_path=True``. ``on_missing`` behaves identically (``"raise"``
    default, ``"skip"`` drops unresolved triples into ``summary.missing``).
    """
    _check_as_of(as_of)
    _check_on_missing(on_missing)

    if "data_type" not in manifest.columns or "name" not in manifest.columns:
        raise ManifestError("Manifest is missing required columns: ['data_type', 'name']")

    has_edge = "edge_uuid" in manifest.columns or all(
        c in manifest.columns for c in ("from_path", "to_path", "edge_type")
    )
    if has_edge:
        raise ManifestError(
            "as_of reads are supported for node-routed manifests (path or node_uuid); "
            "edge history is not tracked."
        )

    has_path = "path" in manifest.columns
    has_node = "node_uuid" in manifest.columns
    if has_path and has_node:
        raise ManifestError("Manifest has ambiguous routing columns ['path', 'node_uuid']; provide exactly one.")
    if not (has_path or has_node):
        raise ManifestError("as_of manifest must include a 'path' or 'node_uuid' routing column.")

    manifest = manifest.with_columns(pl.col("data_type").cast(pl.Utf8).str.to_lowercase())
    if has_path:
        return await _as_of_meta_by_path(conn, manifest, as_of=as_of, on_missing=on_missing)
    return await _as_of_meta_by_owner(conn, manifest, as_of=as_of, on_missing=on_missing)


async def resolve_subtree_series_as_of(
    conn,
    *,
    as_of: datetime,
    root_path: str | None = None,
    start_uuid: UUID | None = None,
    rel_path: str | None = None,
    where_conds: list[str] | None = None,
    where_params: list[Any] | None = None,
    data_type: str | None = None,
    name: str | None = None,
) -> pl.DataFrame:
    """As-of counterpart of :func:`energydb.series.resolve_subtree_series_for_read`.

    Resolves the subtree as it existed at knowledge time ``as_of`` straight to
    per-series read meta in one round-trip. The structural walk reads
    ``node_history`` intervals covering ``as_of`` (each node's row then carried
    the path it held at that time) instead of the live ``node`` table; the
    ``series`` catalog still joins by the stable node uuid, so a point moved A→B
    keeps the same series and values while its subtree membership and path label
    reflect group A at an as-of before the move.

    The root is given either as ``root_path`` (the ``/``-joined path valid at
    ``as_of``) or as ``start_uuid`` (+ optional ``rel_path``), whose as-of path
    is derived inline. ``where_conds`` / ``where_params`` qualify the history row
    with alias ``h``. Returns the same column schema as the live resolver
    (``series_id``, ``canonical_unit``, ``timeseries_type``, ``retention``,
    ``node_uuid``, ``data_type``, ``name``, ``path``); an empty frame when the
    subtree did not exist or had no matching series at ``as_of``.
    """
    from energydb.paths import derived_prefix_like  # local: keeps module-import graph shallow

    _check_as_of(as_of)
    if root_path is not None:
        root_cte = "SELECT %s::text AS rp"
        cte_params: list[Any] = [root_path]
    elif start_uuid is not None and rel_path:
        root_cte = f"""
            SELECT (path || '/' || %s) AS rp FROM {P}node_history
            WHERE node_uuid = %s AND valid_from <= %s
              AND (valid_to IS NULL OR valid_to > %s)
        """
        cte_params = [rel_path, start_uuid, as_of, as_of]
    elif start_uuid is not None:
        root_cte = f"""
            SELECT path AS rp FROM {P}node_history
            WHERE node_uuid = %s AND valid_from <= %s
              AND (valid_to IS NULL OR valid_to > %s)
        """
        cte_params = [start_uuid, as_of, as_of]
    else:
        raise ValidationError("resolve_subtree_series_as_of needs root_path or start_uuid.")

    where_parts = list(where_conds or [])
    where_vals: list[Any] = list(where_params or [])
    if data_type:
        where_parts.append("s.data_type = %s")
        where_vals.append(data_type)
    if name:
        where_parts.append("s.name = %s")
        where_vals.append(name)
    where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""

    # The interval predicate selects each node's single row that was current at
    # as_of; the partial unique index ix_node_history_current plus
    # ix_node_history_path_interval makes the covering lookup index-friendly.
    sql = rf"""
        WITH root AS ({root_cte})
        SELECT s.series_id, s.canonical_unit, s.timeseries_type, s.retention,
               s.node_uuid::text AS node_uuid, s.data_type, s.name, h.path AS path
        FROM root r
        JOIN {P}node_history h
          ON h.valid_from <= %s
         AND (h.valid_to IS NULL OR h.valid_to > %s)
         AND (h.path = r.rp
              OR h.path LIKE
                 replace(replace(replace(r.rp, E'\\', E'\\\\'), '%%', E'\\%%'), '_', E'\\_')
                 || '/%%' ESCAPE '\')
        JOIN {P}series s ON s.node_uuid = h.node_uuid
        WHERE TRUE{where_clause}
    """
    rows = await (
        await conn.execute(sql, [*cte_params, as_of, as_of, *where_vals])
    ).fetchall()

    return pl.DataFrame(
        [
            {
                "series_id": r[0],
                "canonical_unit": r[1],
                "timeseries_type": r[2],
                "retention": r[3],
                "node_uuid": r[4],
                "data_type": r[5],
                "name": r[6],
                "path": r[7],
            }
            for r in rows
        ],
        schema={
            "series_id": pl.Int64,
            "canonical_unit": pl.Utf8,
            "timeseries_type": pl.Utf8,
            "retention": pl.Utf8,
            "node_uuid": pl.Utf8,
            "data_type": pl.Utf8,
            "name": pl.Utf8,
            "path": pl.Utf8,
        },
    )


async def get_node_history(conn, node_uuid: UUID) -> list[NodeHistoryRow]:
    """Return a node's full structural timeline, oldest interval first.

    Each row is ``{node_uuid, node_type, name, parent_uuid, path, valid_from,
    valid_to}`` with timezone-aware datetimes; ``valid_to`` is None on the
    current interval. Empty list when the node has no history (never created /
    already purged).
    """
    rows = await (
        await conn.execute(
            f"""
            SELECT node_uuid, node_type, name, parent_uuid, path, valid_from, valid_to
            FROM {P}node_history
            WHERE node_uuid = %s
            ORDER BY valid_from
            """,
            (node_uuid,),
        )
    ).fetchall()
    return [
        {
            "node_uuid": r[0],
            "node_type": r[1],
            "name": r[2],
            "parent_uuid": r[3],
            "path": r[4],
            "valid_from": r[5],
            "valid_to": r[6],
        }
        for r in rows
    ]


__all__ = [
    "NodeHistoryRow",
    "backfill_node_history",
    "close_node_interval",
    "get_node_history",
    "open_node_interval",
    "replace_subtree_intervals",
    "resolve_manifest_as_of",
    "resolve_node_uuid_as_of",
    "resolve_path_as_of",
    "resolve_subtree_series_as_of",
    "transaction_now",
]
