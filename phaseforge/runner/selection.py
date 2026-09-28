"""Pure selection resolution for the sweep runner (facet-filter model).

A *selection* is the set of manifest cells a sweep will run. Cells are
identified by their ``(task, name)`` identity — displayed as
``name@task`` — because the five-task protocol replicates every method name
once per task, and a bare name therefore names a *family* of cells, not one
cell.

The user request (:class:`SelectionSpec`) is resolved by
:func:`resolve_selection` into a :class:`SelectionResult`: the matched cells
in manifest order, plus the per-token matches for auditing. Resolution is
pure — no I/O, no globals — so every rule below is unit-testable and the CLI
stays thin.

Grammar for ``--methods`` tokens:

* ``<index>``     — the manifest row with that index (legacy form).
* ``<name>``      — every cell carrying that name, intersected with the
  ``--tasks`` facet filter when one is given. Multi-task matches are the
  *intended* semantics here, not an ambiguity.
* ``<name>@<task>`` — one exact cell.

Error policy: every failure mode is loud and actionable — close-match
suggestions for unknown names, the valid task list for unknown tasks, an
explicit diagnosis for empty intersections — and all of them raise before
any step executes.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass

from phaseforge.runner.protocol import Method, Protocol, ProtocolError


@dataclass(frozen=True)
class SelectionSpec:
    """The raw user request, exactly as expressed on the command line."""

    method_tokens: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    """The resolved cell set plus how each token matched (for audit)."""

    methods: tuple[Method, ...]
    token_matches: tuple[tuple[str, tuple[Method, ...]], ...] = ()
    tasks: tuple[str, ...] = ()


def _canonical(value: str) -> str:
    return value.strip().lower()


def effective_task(method: Method, protocol: Protocol) -> str | None:
    """The task a row belongs to for selection purposes.

    Rows that carry no task inherit the protocol-level task name, unless that
    name is the multi-task placeholder ``"all"`` — a task-less row under an
    ``"all"`` protocol belongs to no concrete task.
    """
    if method.task is not None:
        return method.task
    nominal = protocol.task.strip()
    if nominal and nominal.lower() != "all":
        return nominal
    return None


def _cell_label(method: Method, protocol: Protocol) -> str:
    task = effective_task(method, protocol)
    return f"{method.name}@{task}" if task is not None else method.name


def _distinct_tasks(methods: list[Method], protocol: Protocol) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in methods:
        task = effective_task(m, protocol)
        label = task if task is not None else "-"
        if _canonical(label) not in seen:
            seen.add(_canonical(label))
            out.append(label)
    return out


def _suggest_name(token: str, protocol: Protocol) -> str:
    names = sorted({m.name for m in protocol.methods})
    close = difflib.get_close_matches(token, names, n=3, cutoff=0.6)
    if close:
        return f" Did you mean: {', '.join(close)}?"
    return f" Valid names: {names}."


def resolve_selection(protocol: Protocol, spec: SelectionSpec) -> SelectionResult:
    """Resolve a :class:`SelectionSpec` against a loaded protocol.

    Rules, in order:

    1. ``tasks`` (the ``--tasks`` facet) must name known tasks
       (case-insensitive); it then keeps only the rows whose effective task
       is in the filter.
    2. ``method_tokens`` resolve independently and union: an index selects
       one row and must lie inside the task filter; ``name@task`` selects
       one exact cell; a bare name selects all cells with that name inside
       the task filter.
    3. With no tokens, the task filter alone defines the selection; with
       neither, every row is selected.

    Raises :class:`ProtocolError` on any unknown reference, any task-filter
    violation, and any resolution that ends with zero cells.
    """
    task_filter: set[str] | None = None
    resolved_tasks: list[str] = []
    if spec.tasks:
        known = protocol.known_tasks
        if not known:
            raise ProtocolError(
                f"--tasks {list(spec.tasks)} was given, but protocol "
                f"{protocol.name!r} declares no task dimension."
            )
        known_by_canonical = {_canonical(t): t for t in known}
        for raw in spec.tasks:
            display = known_by_canonical.get(_canonical(raw))
            if display is None:
                raise ProtocolError(
                    f"Unknown task {raw!r}. Valid tasks: {list(known)}."
                )
            if _canonical(display) not in {_canonical(t) for t in resolved_tasks}:
                resolved_tasks.append(display)
        task_filter = {_canonical(t) for t in resolved_tasks}

    def in_task_filter(method: Method) -> bool:
        if task_filter is None:
            return True
        task = effective_task(method, protocol)
        return task is not None and _canonical(task) in task_filter

    token_matches: list[tuple[str, tuple[Method, ...]]] = []
    selected_keys: set[str] = set()

    if spec.method_tokens:
        for token in spec.method_tokens:
            matched = _resolve_token(protocol, token, task_filter, in_task_filter)
            token_matches.append((token, tuple(matched)))
            selected_keys.update(m.phase_key for m in matched)
        methods = tuple(m for m in protocol.methods if m.phase_key in selected_keys)
    else:
        methods = tuple(m for m in protocol.methods if in_task_filter(m))

    if not methods:
        raise ProtocolError(
            "Selection resolved to 0 cells "
            f"(tokens={list(spec.method_tokens)}, tasks={list(spec.tasks)}). "
            "Widen the filters or check the spelling against "
            "`phaseforge-sweep --list`."
        )
    return SelectionResult(
        methods=methods,
        token_matches=tuple(token_matches),
        tasks=tuple(resolved_tasks),
    )


def _resolve_token(
    protocol: Protocol,
    token: str,
    task_filter: set[str] | None,
    in_task_filter: Callable[[Method], bool],
) -> list[Method]:
    """Resolve one ``--methods`` token to its matched rows (never empty)."""
    rows = list(protocol.methods)

    # isdecimal, not isdigit: superscript digits (e.g. '²') satisfy isdigit
    # but crash int(), and that ValueError is not a ProtocolError — the
    # token must land in the unknown-name diagnostic instead.
    if token.isdecimal():
        method = protocol.method_by_index(int(token))
        if method is None:
            raise ProtocolError(
                f"Unknown method index {token!r}. Valid indices: "
                f"{[r.index for r in rows]}."
            )
        if not in_task_filter(method):
            raise ProtocolError(
                f"Method index {token} resolves to {_cell_label(method, protocol)}, "
                "which is outside the selected tasks. Drop --tasks or use an "
                "index from the wanted task."
            )
        return [method]

    if "@" in token:
        name, _, task = token.partition("@")
        wanted = _canonical(task)
        matched = [
            r
            for r in rows
            if r.name == name
            and effective_task(r, protocol) is not None
            and _canonical(effective_task(r, protocol)) == wanted
        ]
        if not matched:
            same_name = [r for r in rows if r.name == name]
            if not same_name:
                raise ProtocolError(f"Unknown method {name!r}.{_suggest_name(name, protocol)}")
            available = _distinct_tasks(same_name, protocol)
            raise ProtocolError(
                f"Method {name!r} exists but has no cell on task {task!r}; "
                f"it exists on: {available}."
            )
        if task_filter is not None and wanted not in task_filter:
            raise ProtocolError(
                f"{token!r} selects a cell outside the --tasks filter; remove "
                f"{token!r} or widen the filter."
            )
        return matched

    matched = [r for r in rows if r.name == token and in_task_filter(r)]
    if not matched:
        same_name = [r for r in rows if r.name == token]
        if same_name:
            available = _distinct_tasks(same_name, protocol)
            raise ProtocolError(
                f"Method {token!r} has no cells inside the selected tasks; "
                f"it exists on: {available}."
            )
        raise ProtocolError(f"Unknown method {token!r}.{_suggest_name(token, protocol)}")
    return matched


def format_selection_table(result: SelectionResult, protocol: Protocol) -> str:
    """One line per resolved cell, for the pre-execution preview."""
    lines: list[str] = []
    for m in result.methods:
        stages = ",".join(str(s) for s in m.stages)
        mode = m.evaluate_mode if m.evaluate else "no"
        lines.append(
            f"  {m.index:>4}  {_cell_label(m, protocol):<36} stages {stages:<6} eval={mode}"
        )
    return "\n".join(lines)
