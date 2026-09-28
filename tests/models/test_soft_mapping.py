"""CPU-only tests for the soft phase->expert mapping construction."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phaseforge.models.components.soft_mapping import (
    build_hierarchical_uniform_mapping,
    build_prototype_softmax_mapping,
    build_soft_mapping,
    validate_soft_mapping,
)


class TestValidateSoftMapping:
    def test_right_stochastic_rows_pass(self) -> None:
        mapping = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        validate_soft_mapping(mapping)

    def test_negative_entries_rejected(self) -> None:
        mapping = torch.tensor([[0.5, 0.5], [1.0, -1e-6]])
        with pytest.raises(ValueError, match="negative"):
            validate_soft_mapping(mapping)

    def test_rows_not_summing_to_one_rejected(self) -> None:
        mapping = torch.tensor([[0.4, 0.5], [1.0, 0.0]])
        with pytest.raises(ValueError, match="sum to 1"):
            validate_soft_mapping(mapping)

    def test_non_finite_rejected(self) -> None:
        mapping = torch.tensor([[float("nan"), 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="non-finite"):
            validate_soft_mapping(mapping)

    def test_non_2d_rejected(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            validate_soft_mapping(torch.zeros(3))

    def test_matches_single_expert_row_with_epsilon(self) -> None:
        # A single-expert row legitimately has ~1.0; a bare 1.0/1.0 sums
        # exactly, and softmax rows are within atol after construction.
        validate_soft_mapping(torch.tensor([[1.0, 0.0]]))


class TestHierarchicalUniform:
    def test_eight_experts_six_phases_layout(self) -> None:
        # E=8 > P=6: base_k=1, rem=2 -> phases 0,1 own two experts each.
        mapping = build_hierarchical_uniform_mapping(num_phases=6, num_experts=8)
        assert mapping.shape == (6, 8)
        assert torch.allclose(
            mapping[0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        assert torch.allclose(
            mapping[1], torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        )
        assert torch.allclose(
            mapping[2], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        )
        assert torch.allclose(
            mapping[5], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def test_equal_phases_experts_is_identity(self) -> None:
        mapping = build_hierarchical_uniform_mapping(num_phases=3, num_experts=3)
        assert torch.equal(mapping, torch.eye(3))

    def test_deterministic(self) -> None:
        first = build_hierarchical_uniform_mapping(6, 8)
        second = build_hierarchical_uniform_mapping(6, 8)
        assert torch.equal(first, second)

    def test_invalid_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_hierarchical_uniform_mapping(num_phases=0, num_experts=8)


class TestPrototypeSoftmax:
    def _centroids(self) -> torch.Tensor:
        return F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), dim=-1)

    def _prototypes(self) -> torch.Tensor:
        return F.normalize(
            torch.tensor([[0.9, 0.1], [0.1, 0.9], [-0.1, 0.9], [0.0, -0.9]]), dim=-1
        )

    def test_rows_sum_to_one_and_finite(self) -> None:
        mapping = build_prototype_softmax_mapping(self._centroids(), self._prototypes())
        assert mapping.shape == (3, 4)
        assert torch.allclose(mapping.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert torch.isfinite(mapping).all()

    def test_lower_temperature_sharpens_rows(self) -> None:
        centroids = self._centroids()
        prototypes = self._prototypes()
        soft = build_prototype_softmax_mapping(centroids, prototypes, temperature=4.0)
        sharp = build_prototype_softmax_mapping(centroids, prototypes, temperature=0.1)
        soft_entropy = -(soft * torch.log(soft.clamp_min(1e-9))).sum(dim=-1).mean()
        sharp_entropy = -(sharp * torch.log(sharp.clamp_min(1e-9))).sum(dim=-1).mean()
        assert sharp_entropy < soft_entropy

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Dimension mismatch"):
            build_prototype_softmax_mapping(
                torch.zeros(3, 4), torch.zeros(3, 5)
            )

    def test_non_positive_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            build_prototype_softmax_mapping(self._centroids(), self._prototypes(), temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            build_prototype_softmax_mapping(
                self._centroids(), self._prototypes(), temperature=float("nan")
            )


class TestDispatch:
    def test_hierarchical_uniform_dispatch(self) -> None:
        mapping = build_soft_mapping("hierarchical_uniform", num_phases=6, num_experts=8)
        assert mapping.shape == (6, 8)

    def test_prototype_softmax_dispatch(self) -> None:
        mapping = build_soft_mapping(
            "prototype_softmax",
            phase_centroids=F.normalize(torch.randn(3, 4), dim=-1),
            expert_prototypes=F.normalize(torch.randn(5, 4), dim=-1),
        )
        assert mapping.shape == (3, 5)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown soft-mapping mode"):
            build_soft_mapping("mystery", num_phases=6, num_experts=8)

    def test_missing_requirements_raise(self) -> None:
        with pytest.raises(ValueError, match="num_phases and num_experts"):
            build_soft_mapping("hierarchical_uniform")
        with pytest.raises(ValueError, match="phase_centroids and expert_prototypes"):
            build_soft_mapping("prototype_softmax")