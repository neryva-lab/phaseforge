"""Unit tests for the pure selection resolver (facet-filter model).

All fixtures are in-memory Protocols — resolution must never touch the
filesystem. The multi-task fixture repeats final method identities across
tasks; the single-task fixture uses task-less rows under protocol task
``"Lift"``.
"""

from __future__ import annotations

import pytest

from phaseforge.runner.protocol import Method, Protocol, ProtocolError
from phaseforge.runner.selection import (
    SelectionSpec,
    effective_task,
    format_selection_table,
    resolve_selection,
)


def _m(
    index: int,
    name: str,
    task: str | None = None,
    stages: tuple[int, ...] = (1,),
    evaluate: bool = True,
) -> Method:
    return Method(
        index=index,
        name=name,
        role="test",
        model="final_aligned_bc" if name == "bc" else name,
        data="common",
        stages=stages,
        stage2_source=None,
        evaluate=evaluate,
        task=task,
    )


def _proto(methods: list[Method], task: str = "Lift") -> Protocol:
    return Protocol(
        name="test",
        task=task,
        description="synthetic",
        seeds=(42,),
        defaults=(),
        methods=tuple(methods),
    )


@pytest.fixture
def multi() -> Protocol:
    """Multi-task shape: every name replicated per task, protocol task 'all'."""
    return _proto(
        [
            _m(1, "precision_residual_phaseforge", task="Lift", stages=(1, 2)),
            _m(2, "bc", task="Lift"),
            _m(3, "precision_residual_phaseforge", task="Can", stages=(1, 2)),
            _m(4, "bc", task="Can"),
            _m(5, "precision_residual_phaseforge", task="Transport", stages=(1, 2)),
            _m(6, "bc", task="Transport"),
        ],
        task="all",
    )


@pytest.fixture
def single() -> Protocol:
    """Single-task shape: task-less rows under a single-task protocol."""
    return _proto(
        [
            _m(1, "precision_residual_phaseforge", stages=(1, 2)),
            _m(2, "bc"),
            _m(21, "precision_residual_plain_encoder", stages=(2,)),
        ],
        task="Lift",
    )


class TestBareName:
    def test_selects_every_task_for_that_name(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("precision_residual_phaseforge",))
        )
        assert [m.phase_key for m in result.methods] == [
            "Lift/precision_residual_phaseforge",
            "Can/precision_residual_phaseforge",
            "Transport/precision_residual_phaseforge",
        ]

    def test_name_intersected_with_task_filter(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("bc",), tasks=("Can",))
        )
        assert [m.phase_key for m in result.methods] == ["Can/bc"]

    def test_task_matching_is_case_insensitive(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("bc",), tasks=("lift",))
        )
        assert [m.phase_key for m in result.methods] == ["Lift/bc"]

    def test_name_with_no_cells_in_task_filter_is_loud(self) -> None:
        protocol = _proto(
            [
                _m(1, "precision_residual_phaseforge", task="Lift"),
                _m(2, "bc", task="Can"),
            ],
            task="all",
        )
        with pytest.raises(ProtocolError, match=r"exists on: \['Lift'\]"):
            resolve_selection(
                protocol,
                SelectionSpec(
                    method_tokens=("precision_residual_phaseforge",), tasks=("Can",)
                ),
            )

    def test_unknown_name_gets_close_match_suggestion(self, single: Protocol) -> None:
        with pytest.raises(ProtocolError, match="precision_residual_phaseforg"):
            resolve_selection(
                single, SelectionSpec(method_tokens=("precision_residual_phaseforg",))
            )


class TestExplicitCell:
    def test_name_at_task_selects_one_cell(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("precision_residual_phaseforge@Can",))
        )
        assert [m.phase_key for m in result.methods] == [
            "Can/precision_residual_phaseforge"
        ]

    def test_name_at_task_case_insensitive(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("precision_residual_phaseforge@can",))
        )
        assert [m.phase_key for m in result.methods] == [
            "Can/precision_residual_phaseforge"
        ]

    def test_wrong_task_lists_available_tasks(self, multi: Protocol) -> None:
        with pytest.raises(ProtocolError, match="no cell on task 'Square'"):
            resolve_selection(
                multi,
                SelectionSpec(method_tokens=("precision_residual_phaseforge@Square",)),
            )

    def test_name_at_task_outside_task_filter_is_rejected(self, multi: Protocol) -> None:
        with pytest.raises(ProtocolError, match="outside the --tasks filter"):
            resolve_selection(
                multi,
                SelectionSpec(
                    method_tokens=("precision_residual_phaseforge@Lift",), tasks=("Can",)
                ),
            )

    def test_name_at_task_on_single_task_manifest(self, single: Protocol) -> None:
        result = resolve_selection(
            single, SelectionSpec(method_tokens=("bc@Lift",))
        )
        assert [m.phase_key for m in result.methods] == ["bc"]


class TestIndexToken:
    def test_index_selects_exact_row(self, multi: Protocol) -> None:
        result = resolve_selection(multi, SelectionSpec(method_tokens=("4",)))
        assert [m.phase_key for m in result.methods] == ["Can/bc"]

    def test_index_outside_task_filter_is_loud(self, multi: Protocol) -> None:
        with pytest.raises(ProtocolError, match=r"index 4 resolves to bc@Can"):
            resolve_selection(
                multi, SelectionSpec(method_tokens=("4",), tasks=("Lift",))
            )

    def test_unknown_index_lists_valid(self, multi: Protocol) -> None:
        with pytest.raises(ProtocolError, match="Unknown method index '99'"):
            resolve_selection(multi, SelectionSpec(method_tokens=("99",)))

    def test_superscript_digit_token_is_a_clean_error(self, multi: Protocol) -> None:
        # '²' satisfies str.isdigit but int('²') raises a plain ValueError;
        # the resolver must route it to the unknown-method diagnostic.
        with pytest.raises(ProtocolError, match="Unknown method '²'"):
            resolve_selection(multi, SelectionSpec(method_tokens=("²",)))


class TestTaskFacet:
    def test_tasks_alone_select_every_method_on_those_tasks(self, multi: Protocol) -> None:
        result = resolve_selection(multi, SelectionSpec(tasks=("Can", "Transport")))
        assert [m.phase_key for m in result.methods] == [
            "Can/precision_residual_phaseforge",
            "Can/bc",
            "Transport/precision_residual_phaseforge",
            "Transport/bc",
        ]

    def test_unknown_task_lists_valid_tasks(self, multi: Protocol) -> None:
        with pytest.raises(ProtocolError, match=r"Unknown task 'square'. Valid tasks"):
            resolve_selection(multi, SelectionSpec(tasks=("square",)))

    def test_single_task_manifest_tasks_via_protocol_task(self, single: Protocol) -> None:
        result = resolve_selection(single, SelectionSpec(tasks=("lift",)))
        assert len(result.methods) == 3

    def test_task_dimensionless_manifest_rejects_tasks(self) -> None:
        protocol = _proto([_m(1, "precision_residual_phaseforge")], task="all")
        with pytest.raises(ProtocolError, match="declares no task dimension"):
            resolve_selection(protocol, SelectionSpec(tasks=("Lift",)))


class TestCompositionAndOrder:
    def test_tokens_union_and_dedup_in_manifest_order(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi,
            SelectionSpec(
                method_tokens=(
                    "bc@Can",
                    "precision_residual_phaseforge",
                    "2",
                    "precision_residual_phaseforge",
                )
            ),
        )
        assert [m.phase_key for m in result.methods] == [
            "Lift/precision_residual_phaseforge",
            "Lift/bc",
            "Can/precision_residual_phaseforge",
            "Can/bc",
            "Transport/precision_residual_phaseforge",
        ]
        # Per-token audit trail keeps duplicates of the user's request.
        assert [token for token, _ in result.token_matches] == [
            "bc@Can",
            "precision_residual_phaseforge",
            "2",
            "precision_residual_phaseforge",
        ]

    def test_no_filters_selects_everything(self, multi: Protocol) -> None:
        result = resolve_selection(multi, SelectionSpec())
        assert len(result.methods) == len(multi.methods)

    def test_parity_with_legacy_select_methods_on_single_task(
        self, single: Protocol
    ) -> None:
        legacy = single.select_methods(["precision_residual_phaseforge", "21"])
        result = resolve_selection(
            single,
            SelectionSpec(method_tokens=("precision_residual_phaseforge", "21")),
        )
        assert [m.phase_key for m in result.methods] == [m.phase_key for m in legacy]


class TestHelpers:
    def test_effective_task_row_task_wins(self, multi: Protocol) -> None:
        assert effective_task(multi.methods[0], multi) == "Lift"

    def test_effective_task_falls_back_to_protocol_task(self, single: Protocol) -> None:
        assert effective_task(single.methods[0], single) == "Lift"

    def test_effective_task_none_under_all_protocol(self) -> None:
        protocol = _proto([_m(1, "precision_residual_phaseforge")], task="all")
        assert effective_task(protocol.methods[0], protocol) is None

    def test_table_renders_cell_lines(self, multi: Protocol) -> None:
        result = resolve_selection(
            multi, SelectionSpec(method_tokens=("precision_residual_phaseforge",))
        )
        table = format_selection_table(result, multi)
        assert "precision_residual_phaseforge@Lift" in table
        assert "stages 1,2" in table
        assert "eval=rollout" in table

    def test_known_tasks_manifest_order_distinct(self, multi: Protocol) -> None:
        assert multi.known_tasks == ("Lift", "Can", "Transport")

    def test_known_tasks_single_task_fallback(self, single: Protocol) -> None:
        assert single.known_tasks == ("Lift",)

    def test_known_tasks_all_placeholder_yields_empty(self) -> None:
        protocol = _proto([_m(1, "precision_residual_phaseforge")], task="all")
        assert protocol.known_tasks == ()
