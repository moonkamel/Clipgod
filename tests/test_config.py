import pytest
import yaml

from core import config as C


def test_load_default_config(project):
    cfg = C.load_config(project / "config.yaml")
    assert cfg["_meta"]["root"] == str(project)
    assert cfg["_meta"]["runtime"] == "local"
    assert C.get(cfg, "scan.windows.min_seconds") == 15
    assert C.get(cfg, "watch.disable_watch") is True          # tolère '-' vs '_'
    assert C.path(cfg, "model") == project / "data" / "model.pkl"


def test_env_overrides_are_typed(project):
    env = {"CLIPGOD__SCAN__WHISPER__MODEL": "medium", "CLIPGOD__PUBLISH__MAX_POSTS_PER_DAY": "4",
           "CLIPGOD__WATCH__DISABLE_WATCH": "false"}
    cfg = C.load_config(project / "config.yaml", environ=env)
    assert cfg["scan"]["whisper"]["model"] == "medium"
    assert cfg["publish"]["max_posts_per_day"] == 4
    assert cfg["watch"]["disable-watch"] is False            # clé existante réutilisée


def test_dotted_overrides(project):
    cfg = C.load_config(project / "config.yaml", overrides={"scan.top_k_moments": 5, "render": {"fps": 60}})
    assert cfg["scan"]["top_k_moments"] == 5
    assert cfg["render"]["fps"] == 60 and cfg["render"]["width"] == 1080


def test_validation_collects_errors(project):
    raw = yaml.safe_load((project / "config.yaml").read_text())
    raw["scan"]["windows"]["min_seconds"] = 60
    raw["render"]["layout"]["gameplay_ratio"] = 0.8
    (project / "config.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(C.ConfigError) as exc:
        C.load_config(project / "config.yaml")
    assert "min_seconds" in str(exc.value) and "gameplay_ratio" in str(exc.value)


def test_missing_config(tmp_path):
    with pytest.raises(C.ConfigError):
        C.load_config(tmp_path / "nope.yaml")


def test_dotenv_and_secret(project, monkeypatch):
    (project / ".env").write_text('YOUTUBE_API_KEY="abc123"\n# comment\n')
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    cfg = C.load_config(project / "config.yaml")
    assert C.get_secret(cfg, "discover.env.youtube_api_key") == "abc123"
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    with pytest.raises(C.ConfigError, match="TWITCH_CLIENT_ID"):
        C.get_secret(cfg, "discover.env.twitch_client_id")


def test_job_config_subset(project):
    cfg = C.load_config(project / "config.yaml")
    sub = C.job_config(cfg)
    assert set(sub) >= {"scan", "render"} and "publish" not in sub and "_meta" not in sub
