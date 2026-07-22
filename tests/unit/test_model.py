import torch


def test_transformer_ranker_forward_contract():
    from tensoralpha.models import TransformerConfig, TransformerRanker

    config = TransformerConfig(
        input_dim=12,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    model = TransformerRanker(config).eval()
    features = torch.randn(7, 20, 12)

    first = model(features)
    second = model(features)

    assert first.shape == (7,)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_transformer_ranker_rejects_wrong_feature_width():
    from tensoralpha.models import TransformerConfig, TransformerRanker

    model = TransformerRanker(TransformerConfig(input_dim=8, d_model=16, nhead=4))

    try:
        model(torch.randn(2, 5, 7))
    except ValueError as error:
        assert "input_dim" in str(error)
    else:
        raise AssertionError("wrong feature width was accepted")
