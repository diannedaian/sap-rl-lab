import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib.common.maskable.distributions import MaskableCategorical  # noqa: E402

from sap_rl_lab.stable_masking import FreshMaskableCategorical  # noqa: E402


def test_reproduces_stale_probability_failure_without_hiding_validation():
    old = MaskableCategorical(logits=torch.zeros(2, 139), validate_args=True)
    old.probs = old.probs * 1.00002
    with pytest.raises(ValueError, match="Simplex"):
        old.apply_masking(torch.ones(2, 139, dtype=torch.bool))
    repaired = FreshMaskableCategorical(logits=torch.zeros(2, 139), validate_args=True)
    repaired.probs = repaired.probs * 1.00002
    repaired.apply_masking(torch.ones(2, 139, dtype=torch.bool))
    assert repaired._validate_args
    assert torch.allclose(repaired.probs.sum(-1), torch.ones(2), atol=1e-7)


def test_invalid_logits_and_empty_masks_still_fail():
    with pytest.raises(ValueError):
        FreshMaskableCategorical(logits=torch.tensor([[0.0, float("nan")]]))
    d = FreshMaskableCategorical(logits=torch.zeros(1, 139))
    with pytest.raises(ValueError, match="no legal action"):
        d.apply_masking(torch.zeros(1, 139, dtype=torch.bool))


def test_masks_remain_exact_and_gradients_finite_after_repeated_remasking():
    generator = torch.Generator().manual_seed(67)
    logits = (torch.randn(256, 139, generator=generator) * 12).requires_grad_()
    masks = torch.rand(256, 139, generator=generator) > 0.65
    masks[:, 0] = True
    d = FreshMaskableCategorical(logits=logits)
    for _ in range(5):
        d.apply_masking(masks)
        assert (d.probs[~masks] == 0).all()
        assert torch.isfinite(d.entropy()).all()
        samples = d.sample((10,))
        assert masks.unsqueeze(0).expand(10, -1, -1).gather(2, samples.unsqueeze(-1)).all()
        d.apply_masking(None)
    d.apply_masking(masks)
    loss = -d.log_prob(d.sample()).mean() - 0.01 * d.entropy().mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_normal_case_matches_upstream_probabilities():
    logits = torch.tensor([[0.2, 4.0, -3.0, 1.0]])
    mask = [[True, False, True, True]]
    old = MaskableCategorical(logits=logits, masks=mask)
    new = FreshMaskableCategorical(logits=logits, masks=mask)
    assert torch.equal(old.probs, new.probs)
    assert torch.equal(old.entropy(), new.entropy())


@pytest.mark.parametrize("scale", [1.0, 30.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_entropy_bonus_gradient_matches_compact_legal_action_distribution(scale, dtype):
    # Compare against distributions that never contain illegal actions, rather
    # than merely checking that the masked implementation produces finite values.
    generator = torch.Generator().manual_seed(193)
    original = torch.randn(3, 7, generator=generator, dtype=dtype) * scale
    mask = torch.tensor(
        [
            [True, False, True, False, True, False, False],
            [False, False, False, True, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    logits = original.clone().requires_grad_()
    compact_logits = original.clone().requires_grad_()
    masked = FreshMaskableCategorical(logits=logits)
    for _ in range(3):
        masked.apply_masking(mask)
        masked.apply_masking(None)
    masked.apply_masking(mask)
    # Fixed legal actions isolate the gradient calculation from sampling noise.
    actions = mask.to(torch.int64).argmax(dim=1)
    actual_entropy = masked.entropy()
    actual_loss = -masked.log_prob(actions).mean() - 0.01 * actual_entropy.mean()
    reference = [
        torch.distributions.Categorical(logits=row[legal])
        for row, legal in zip(compact_logits, mask)
    ]
    expected_entropy = torch.stack([d.entropy() for d in reference])
    expected_loss = -torch.stack([d.log_prob(torch.tensor(0)) for d in reference]).mean()
    expected_loss -= 0.01 * expected_entropy.mean()
    actual_loss.backward()
    expected_loss.backward()
    assert torch.allclose(actual_entropy, expected_entropy, atol=2e-6, rtol=1e-6)
    assert torch.allclose(logits.grad, compact_logits.grad, atol=2e-6, rtol=1e-6)
    assert torch.allclose(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]), atol=2e-6)
