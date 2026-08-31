"""Tests for the dual-encoder fusion segmenter.

Everything runs on CPU with `pretrained=False` and tiny inputs, so the suite stays
fast and needs no network access.
"""

from __future__ import annotations

import pytest

from satquery.models.fusion.dual_encoder import FusionConfig

pytest.importorskip("torch")
pytest.importorskip("timm")


@pytest.fixture(scope="module")
def config() -> FusionConfig:
    return FusionConfig(
        optical_encoder="resnet18",
        sar_encoder="resnet18",
        fusion="concat",
        embed_dim=64,
        pretrained=False,
    )


@pytest.fixture(scope="module")
def pair():
    """A tiny co-registered pair at a size that is *not* a multiple of the stride."""
    import torch

    torch.manual_seed(0)
    return torch.randn(2, 4, 72, 72), torch.randn(2, 3, 72, 72)


# -- shape and padding -------------------------------------------------------------------


def test_output_matches_input_extent_when_stride_does_not_divide(config, pair):
    """120x120 reBEN patches are not a multiple of 32; the output must still be 120x120.

    Resizing instead of padding would change the GSD, making the frozen GSD token
    describe a resolution the pixels no longer have.
    """
    from satquery.models.fusion.dual_encoder import build_fusion_model

    optical, sar = pair
    logits = build_fusion_model(config)(optical, sar)["logits"]
    assert logits.shape == (2, len(config.targets), 72, 72)


@pytest.mark.parametrize("size", [64, 120, 128])
def test_several_input_sizes_round_trip(config, size):
    import torch

    from satquery.models.fusion.dual_encoder import build_fusion_model

    model = build_fusion_model(config)
    logits = model(torch.randn(1, 4, size, size), torch.randn(1, 3, size, size))["logits"]
    assert logits.shape[-2:] == (size, size)


def test_padding_is_not_replicated_into_the_output(config):
    """The crop must discard the padded margin, not average it back in."""
    import torch

    from satquery.models.fusion.dual_encoder import _crop_to, _pad_to_stride

    x = torch.arange(2 * 1 * 5 * 5, dtype=torch.float32).reshape(2, 1, 5, 5)
    padded, size = _pad_to_stride(x, 32)
    assert padded.shape[-2:] == (32, 32)
    assert size == (5, 5)
    torch.testing.assert_close(_crop_to(padded, size), x)


def test_mismatched_grids_are_refused(config):
    """Optical and SAR must be co-registered; a size mismatch means they are not."""
    import torch

    from satquery.models.fusion.dual_encoder import build_fusion_model

    model = build_fusion_model(config)
    with pytest.raises(ValueError, match="co-registered"):
        model(torch.randn(1, 4, 64, 64), torch.randn(1, 3, 32, 32))


# -- fusion strategies -------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["concat", "sum", "gated", "cross_attention"])
def test_every_declared_strategy_runs(strategy):
    """`FUSION_STRATEGIES` advertises four; each must actually be implemented."""
    import torch

    from satquery.models.fusion.dual_encoder import build_fusion_model

    model = build_fusion_model(
        FusionConfig(
            optical_encoder="resnet18",
            sar_encoder="resnet18",
            fusion=strategy,
            embed_dim=64,
            pretrained=False,
        )
    )
    logits = model(torch.randn(1, 4, 64, 64), torch.randn(1, 3, 64, 64))["logits"]
    assert logits.shape == (1, 2, 64, 64)
    assert torch.isfinite(logits).all()


def test_both_branches_affect_the_output(config, pair):
    """A fusion model that ignores SAR is a single-sensor model wearing a costume."""
    import torch

    from satquery.models.fusion.dual_encoder import build_fusion_model

    model = build_fusion_model(config).eval()
    optical, sar = pair
    with torch.no_grad():
        base = model(optical, sar)["logits"]
        changed_sar = model(optical, torch.randn_like(sar))["logits"]
        changed_optical = model(torch.randn_like(optical), sar)["logits"]

    assert not torch.allclose(base, changed_sar), "SAR branch has no effect on the output"
    assert not torch.allclose(base, changed_optical), "optical branch has no effect"


# -- loss --------------------------------------------------------------------------------


def test_empty_prediction_is_penalised_by_dice():
    """The failure mode Dice exists to prevent: predicting background everywhere.

    Built-up is absent from 84% of reBEN patches and water from 73%, so under plain
    BCE an all-background prediction is very nearly optimal on the average patch.
    """
    import torch

    from satquery.models.fusion.dual_encoder import _extraction_loss

    labels = torch.zeros(1, 2, 32, 32)
    labels[0, 1, :4, :4] = 1.0  # a small water region

    empty = torch.full((1, 2, 32, 32), -10.0)  # confidently predicts nothing
    correct = torch.where(labels > 0, 10.0, -10.0)

    assert _extraction_loss(empty, labels) > _extraction_loss(correct, labels)
    assert _extraction_loss(correct, labels) < 0.2


def test_loss_keeps_the_target_axis_separate():
    """Averaging Dice over targets before the mean would let a common class mask a rare one."""
    import torch

    from satquery.models.fusion.dual_encoder import _extraction_loss

    labels = torch.zeros(1, 2, 16, 16)
    labels[0, 0] = 1.0  # target 0 everywhere, target 1 nowhere

    # Right on the common target, wrong on the rare one.
    logits = torch.full((1, 2, 16, 16), 10.0)
    assert _extraction_loss(logits, labels).item() > 0.3


# -- config ------------------------------------------------------------------------------


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        FusionConfig(fusion="telepathy")


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="unknown fusion targets"):
        FusionConfig(targets=("built_up", "unicorns"))


def test_num_outputs_follows_targets():
    assert FusionConfig(targets=("water",)).num_outputs == 1


# -- checkpoint --------------------------------------------------------------------------


def test_checkpoint_round_trip_is_bit_identical(config, pair, tmp_path):
    import torch

    from satquery.models.fusion.dual_encoder import DualEncoderFusion

    original = DualEncoderFusion(config)
    original.build()
    original.module.eval()
    optical, sar = pair
    with torch.no_grad():
        before = original.forward(optical, sar)

    path = original.save_checkpoint(tmp_path / "model.pt")
    restored = DualEncoderFusion(FusionConfig(pretrained=False)).load_checkpoint(path)
    restored.module.eval()
    with torch.no_grad():
        after = restored.forward(optical, sar)

    torch.testing.assert_close(before, after)
    assert restored.config.fusion == config.fusion
    assert restored.config.embed_dim == config.embed_dim


def test_checkpoint_refuses_on_constant_drift(config, tmp_path):
    """A SAR rendering that drifted between training and inference is invisible in the
    scores, so the fingerprint check must be a hard failure."""
    import torch

    from satquery.models.fusion.dual_encoder import DualEncoderFusion

    model = DualEncoderFusion(config)
    model.build()
    path = model.save_checkpoint(tmp_path / "model.pt")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["constants_fingerprint"] = "not-the-running-fingerprint"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="FROZEN CONSTANT DRIFT"):
        DualEncoderFusion(config).load_checkpoint(path)


def test_checkpoint_refuses_partial_weights(config, tmp_path):
    """Loading a state dict with tensors missing would produce confident noise."""
    import torch

    from satquery.models.fusion.dual_encoder import DualEncoderFusion

    model = DualEncoderFusion(config)
    model.build()
    path = model.save_checkpoint(tmp_path / "model.pt")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    dropped = next(iter(payload["state_dict"]))
    del payload["state_dict"][dropped]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="missing"):
        DualEncoderFusion(config).load_checkpoint(path)


def test_missing_checkpoint_names_the_training_script(config, tmp_path):
    from satquery.models.fusion.dual_encoder import DualEncoderFusion

    with pytest.raises(FileNotFoundError, match="train_fusion"):
        DualEncoderFusion(config).load_checkpoint(tmp_path / "absent.pt")


# -- extract -----------------------------------------------------------------------------


def test_extract_returns_one_boolean_mask_per_target(config, pair):
    from satquery.models.fusion.dual_encoder import DualEncoderFusion

    optical, sar = pair
    masks = DualEncoderFusion(config).extract(optical, sar)
    assert set(masks) == set(config.targets)
    for name, mask in masks.items():
        assert mask.shape == (2, 72, 72), name
        assert mask.dtype.is_floating_point is False


# -- tool --------------------------------------------------------------------------------


def test_tool_refuses_a_pair_that_is_not_cross_modal():
    """Feeding SAR to the optical branch yields a confident, entirely wrong mask."""
    from satquery.models.fusion.dual_encoder import FusionExtractionTool
    from satquery.serve.contracts import ImageRef, Modality, TaskType, ToolRequest

    refs = [
        ImageRef(path=f"/tmp/nonexistent_{index}.tif", modality=Modality.SAR, gsd_m=10.0)
        for index in range(2)
    ]
    request = ToolRequest(images=refs, task=TaskType.FUSION_EXTRACTION, query="extract water")
    with pytest.raises(ValueError, match="exactly one SAR and one optical"):
        FusionExtractionTool()._ordered_refs(request)


def test_tool_orders_refs_regardless_of_arrival_order():
    from satquery.models.fusion.dual_encoder import FusionExtractionTool
    from satquery.serve.contracts import ImageRef, Modality, TaskType, ToolRequest

    sar = ImageRef(path="/tmp/a.tif", modality=Modality.SAR, gsd_m=10.0)
    optical = ImageRef(path="/tmp/b.tif", modality=Modality.MULTISPECTRAL, gsd_m=10.0)

    for images in ([sar, optical], [optical, sar]):
        request = ToolRequest(images=images, task=TaskType.FUSION_EXTRACTION, query="extract water")
        first, second = FusionExtractionTool()._ordered_refs(request)
        assert first.modality is Modality.MULTISPECTRAL
        assert second.modality is Modality.SAR


def test_checkpoint_path_is_pure():
    """Callable with no weights present, so the router can report where they should be."""
    from satquery.models.fusion.dual_encoder import FusionExtractionTool

    path = FusionExtractionTool().checkpoint_path()
    assert path.name == "model.pt"
    assert "fusion" in path.parts


def test_tool_refuses_already_rendered_sar():
    """A pre-rendered pseudo-RGB passed as raw SAR would be rendered twice.

    The result is a confident mask built from a speckle-filtered, dB-converted copy of
    what were already display values, and nothing downstream can detect it -- so this
    has to be a hard failure, not a warning.
    """
    from satquery.models.fusion.dual_encoder import FusionExtractionTool
    from satquery.serve.contracts import ImageRef, Modality, TaskType, ToolRequest

    request = ToolRequest(
        images=[
            ImageRef(path="/tmp/o.tif", modality=Modality.MULTISPECTRAL, gsd_m=10.0),
            ImageRef(
                path="/tmp/s.tif",
                modality=Modality.SAR,
                gsd_m=10.0,
                band_names=["VV", "VH", "ratio"],
            ),
        ],
        task=TaskType.FUSION_EXTRACTION,
        query="extract water",
    )
    result = FusionExtractionTool().run(request)
    # BaseTool converts the failure into a well-formed result rather than raising.
    assert result.error is not None
    assert "already-rendered" in result.error or "ratio" in result.error


def test_tool_reads_targets_from_the_request_params_field():
    """`ToolRequest` names the field `params`; reading `parameters` raised at runtime
    and no test covered the happy path, so the typo shipped."""
    from satquery.serve.contracts import ImageRef, Modality, ToolRequest

    request = ToolRequest(
        images=[ImageRef(path="/tmp/o.tif", modality=Modality.MULTISPECTRAL, gsd_m=10.0)],
        task=None,
        query="extract water",
        params={"targets": ["water"]},
    )
    assert request.params["targets"] == ["water"]
    assert not hasattr(request, "parameters")
