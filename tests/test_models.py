from inf8239_u02.models import build_models

TEXTS = ["win a free prize now", "call to claim cash prize", "free entry win cash",
         "see you at lunch", "ok i will call you later", "are you at home now"] * 5
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"] * 5


def test_three_models_with_tfidf_inside_pipeline():
    models = build_models()
    assert set(models) == {"baseline_dummy", "tfidf_nb", "tfidf_logreg"}
    assert all(list(m.named_steps) == ["tfidf", "model"] for m in models.values())


def test_pipelines_predict_one_label_per_text():
    for model in build_models().values():
        model.fit(TEXTS, LABELS)
        assert len(model.predict(["free cash now", "see you later"])) == 2
