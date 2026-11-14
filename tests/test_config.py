import json

from voxflow.config import Config, History, load_config, save_config


def test_defaults_and_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = load_config(p)                      # missing -> defaults + file created
    assert cfg.hotkey == "ctrl+win" and cfg.cleanup_enabled and p.exists()
    cfg.hotkey, cfg.mic_device, cfg.cleanup_enabled = "ctrl+alt+space", 3, False
    save_config(cfg, p)
    again = load_config(p)
    assert (again.hotkey, again.mic_device, again.cleanup_enabled) == ("ctrl+alt+space", 3, False)


def test_corrupt_and_wrong_types(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_config(p) == Config()
    p.write_text(json.dumps({"hotkey": 5, "cleanup_enabled": "yes", "history_limit": 12.0, "bogus": 1,
                             "cleanup_timeout_s": 2, "mic_device": "USB Mic"}), encoding="utf-8")
    c = load_config(p)
    assert c.hotkey == "ctrl+win" and c.cleanup_enabled is True
    assert c.history_limit == 12 and c.cleanup_timeout_s == 2.0 and c.mic_device == "USB Mic"


def test_history_limit_persist(tmp_path):
    p = tmp_path / "h.json"
    h = History(p, limit=3)
    for i in range(5):
        h.add(f"t{i}")
    assert [x["text"] for x in h.snapshot()] == ["t4", "t3", "t2"]
    assert [x["text"] for x in History(p, limit=3).snapshot()] == ["t4", "t3", "t2"]
    h.clear()
    assert History(p).snapshot() == []
