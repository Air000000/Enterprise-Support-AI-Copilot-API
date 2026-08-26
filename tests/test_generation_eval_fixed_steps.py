import experiments.evals.eval_techqa_generation as generation_eval


def test_correctness_judge_uses_fixed_evaluation_steps(monkeypatch):
    observed: dict[str, object] = {}

    class FakeGEval:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.score = 0.8
            self.reason = "correct"

        def measure(self, test_case):
            observed["test_case"] = test_case

    class FakeFaithfulnessMetric:
        def __init__(self, **kwargs):
            self.score = 0.9
            self.reason = "grounded"

        def measure(self, test_case):
            return None

    monkeypatch.setattr(generation_eval, "GEval", FakeGEval)
    monkeypatch.setattr(
        generation_eval,
        "FaithfulnessMetric",
        FakeFaithfulnessMetric,
    )
    monkeypatch.setattr(generation_eval, "_default_judge_model", lambda: object())

    result = generation_eval.judge_techqa_generation(
        question="How do I restart the service?",
        generated_answer="Restart the service.",
        gold_answer="Restart the service.",
        retrieval_context=["Restart the service to recover it."],
    )

    assert "criteria" not in observed
    assert observed["evaluation_steps"] == [
        "Compare the actual output with the expected output and identify any factual contradictions.",
        "Verify that the actual output directly answers the user's input and preserves the key technical claims or instructions in the expected output.",
        "Penalize omissions only when they make the answer materially incorrect or incomplete; do not penalize harmless wording differences.",
        "Give a high score only when the central factual claims are correct and there are no material contradictions.",
    ]
    assert result.correctness_score == 0.8
    assert result.faithfulness_score == 0.9
