import pytest

from sap_rl_lab import midgame_confirmation as mc
from sap_rl_lab.expanded_confirmation import read, save_new


def test_fixed_recipe_and_generator_separation():
    cfg = mc.new_config(timesteps=mc.CANDIDATE_STEPS)
    assert cfg["action_cost"] == 0.005
    assert cfg["swap_cost"] == cfg["success_bonus_max"] == cfg["entropy_coefficient"] == 0
    assert cfg["initialize_from"] == ""
    assert cfg["shop_action_limit_mode"] == "force_battle"
    assert cfg["max_actions_per_turn"] == 30
    assert cfg["gamma"] == 1 and cfg["learning_rate"] == 3e-4
    assert cfg["timesteps"] % (cfg["environments"] * cfg["rollout_steps"]) == 0
    seeds = [g[0] for g in mc.GENERATORS.values()]
    assert len(set(seeds)) == 6
    assert not set(seeds) & set(mc.SEEDS)
    assert [g[2] for g in mc.GENERATORS.values()].count("train") == 3
    assert [g[2] for g in mc.GENERATORS.values()].count("validation") == 1
    assert [g[2] for g in mc.GENERATORS.values()].count("test") == 2


def test_pipeline_orders_all_reloads_before_tests(monkeypatch, tmp_path):
    calls = []
    names = [f"{a}-seed{s}" for s in mc.SEEDS for a in mc.ARMS]
    monkeypatch.setattr(mc, "prepare", lambda p: calls.append("prepare"))
    monkeypatch.setattr(mc, "checked", lambda p: {"generator_configs": mc.GENERATORS})
    monkeypatch.setattr(mc, "check_protocol", lambda p: {"arms": dict.fromkeys(names)})
    monkeypatch.setattr(mc, "batch", lambda p, s, ns: calls.append((s, len(ns))))
    monkeypatch.setattr(mc, "seal", lambda p: calls.append("seal"))
    monkeypatch.setattr(mc, "freeze", lambda p: calls.append("freeze"))
    monkeypatch.setattr(mc, "summarize", lambda p: calls.append("summarize"))
    mc.pipeline(tmp_path)
    assert calls == [
        "prepare",
        ("generator", 6),
        "seal",
        ("arm", 6),
        "freeze",
        ("reload", 6),
        ("test", 6),
        "summarize",
    ]
    assert read(tmp_path / "pipeline_complete.json")["human_readable_report_pending"]


def test_pipeline_failure_stops_without_retry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mc, "prepare", lambda p: None)
    monkeypatch.setattr(mc, "checked", lambda p: {"generator_configs": mc.GENERATORS})

    def fail(p, s, ns):
        calls.append(s)
        raise RuntimeError("failed worker")

    monkeypatch.setattr(mc, "batch", fail)
    with pytest.raises(RuntimeError, match="failed worker"):
        mc.pipeline(tmp_path)
    assert calls == ["generator"]
    assert read(tmp_path / "pipeline_failed.json")["automatic_retry"] is False
    assert not (tmp_path / "pipeline_complete.json").exists()


def test_protocol_checks_frozen_generator_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "checked", lambda p: None)
    monkeypatch.setattr(mc, "verify_inputs", lambda p, d: None)
    save_new(tmp_path / "design.json", {})
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    save_new(
        tmp_path / "protocol.json",
        {
            "design_sha256": mc.file_digest(str(tmp_path / "design.json")),
            "roles": {"test": {"one": {"path": str(model), "sha256": mc.file_digest(str(model))}}},
        },
    )
    mc.check_protocol(tmp_path)
    model.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Frozen generator changed"):
        mc.check_protocol(tmp_path)


def test_batch_only_launches_two_at_once(monkeypatch, tmp_path):
    children = []

    class Child:
        def __init__(self, cmd, cwd):
            self.returncode = None
            assert sum(c.returncode is None for c in children) < 2
            children.append(self)

        def poll(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(mc.subprocess, "Popen", Child)
    mc.batch(tmp_path, "generator", list(mc.GENERATORS))
    assert len(children) == 6


def test_test_stage_requires_verified_reload_not_just_file(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "check_protocol", lambda p: {})
    save_new(tmp_path / "design.json", {})
    save_new(tmp_path / "protocol.json", {})
    model = tmp_path / "model.zip"
    model.write_bytes(b"not loaded because gate must reject first")
    digest = mc.file_digest(str(model))
    save_new(
        tmp_path / "selection.json",
        {
            "protocol_sha256": mc.file_digest(str(tmp_path / "protocol.json")),
            "models": {"one": {"path": str(model), "sha256": digest}},
        },
    )
    save_new(
        tmp_path / "reload-one.json",
        {
            "all_validation_rows_exact": False,
            "model_sha256": digest,
            "selection_sha256": mc.file_digest(str(tmp_path / "selection.json")),
        },
    )
    with pytest.raises(ValueError, match="All frozen-model reloads"):
        mc.inference(tmp_path, "one", "test")
