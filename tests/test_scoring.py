import numpy as np
import pytest

from core import scoring as S


def test_robust_normalize():
    x = S.robust_normalize(np.array([0, 1, 2, 3, 100.0]))
    assert x.min() == 0 and x.max() == 1
    assert np.all(S.robust_normalize(np.ones(5)) == 0)


def test_weighted_scorer_renormalizes_missing_signals():
    sc = S.WeightedScorer()
    X = np.zeros((2, len(S.FEATURES)))
    j = S.FEATURES.index("audio_loudness_peak")
    X[0, j] = 1.0
    # Sans most_replayed / visuel, le pic audio pèse davantage : pas de pénalité pour un signal absent.
    full = sc.score(X, S.FEATURES)
    partial = sc.score(X, S.FEATURES, available={"audio_loudness_peak", "text_hype_keywords"})
    assert partial[0] > full[0] and full[1] == 0
    assert 0 <= partial[0] <= 100


def _bundle(n=60):
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.random((n, len(S.FEATURES)))
    y = 3 + 4 * X[:, S.FEATURES.index("text_hype_keywords")]       # log1p(vues)
    model = GradientBoostingRegressor(n_estimators=50).fit(X, y)
    imp = dict(zip(S.FEATURES, model.feature_importances_, strict=True))
    return {"format": S.MODEL_FORMAT, "features": list(S.FEATURES), "model": model, "weights": imp,
            "n_samples": n, "target": "views_24h", "reference_log_target": sorted(y.tolist()),
            "trained_at": "2026-11-22T00:00:00Z"}


def test_model_scorer_learns_preference(tmp_path):
    import joblib

    path = tmp_path / "model.pkl"
    joblib.dump(_bundle(), path)
    scorer = S.load_scorer({"scan": {"model_path": str(path), "model_blend": 1.0}})
    assert isinstance(scorer, S.ModelScorer) and scorer.alpha == pytest.approx(1.0)
    X = np.zeros((2, len(S.FEATURES)))
    X[0, S.FEATURES.index("text_hype_keywords")] = 1.0
    X[1, S.FEATURES.index("visual_motion")] = 1.0
    s = scorer.score(X, S.FEATURES)
    assert s[0] > s[1]


def test_low_sample_model_has_low_influence():
    scorer = S.ModelScorer(_bundle(10), blend=0.5, full_trust_samples=50)
    assert scorer.alpha == pytest.approx(0.1)


def test_load_scorer_fallbacks(tmp_path):
    assert type(S.load_scorer({"scan": {"model_path": str(tmp_path / "none.pkl")}})) is S.WeightedScorer
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    assert type(S.load_scorer({"scan": {"model_path": str(bad)}})) is S.WeightedScorer


def test_hooks_constraints_and_determinism():
    kw = {"transcript": "I'm gonna lose the cops. No way! NO WAY bro the police are everywhere",
              "signals": {"most_replayed": 0.9, "text_hype_keywords": 0.8}, "keywords": ["no way", "cops"],
              "streamer": "@kaicenat", "n": 3, "max_chars": 60, "seed": "vid:61.0"}
    hooks = S.generate_hooks(**kw)
    assert len(hooks) == 3 and len({h.lower() for h in hooks}) == 3
    assert all(len(h) <= 60 for h in hooks)
    assert hooks == S.generate_hooks(**kw)
    assert any("NO WAY" in h for h in hooks)                       # citation du transcript
    assert all("@" not in h for h in hooks)


def test_hooks_without_transcript_and_french():
    hooks = S.generate_hooks(n=3, lang="fr", seed=1)
    assert len(hooks) == 3 and all(len(h) <= 60 for h in hooks)


def test_build_reason():
    reason = S.build_reason({"audio_loudness_peak": 0.9, "text_hype_keywords": 1.0, "visual_motion": 0.1},
                            S.DEFAULT_WEIGHTS, keywords=["no way", "wtf"])
    assert reason.startswith("mots de hype") and "« no way »" in reason


def test_is_latin_text():
    assert S.is_latin_text("NO WAY bro, c'était énorme !")
    assert not S.is_latin_text("भाई अरे भाई आखी") and not S.is_latin_text("Что это было")
    assert S.is_latin_text("💀 123 !!")                      # pas de lettres : considéré latin


def test_hooks_never_quote_non_latin_transcript():
    hooks = S.generate_hooks(transcript="भाई अरे भाई आखी! भाई!", streamer="ShehzadGaming4K", n=3, seed="x")
    assert len(hooks) == 3 and all(S.is_latin_text(h) for h in hooks)
