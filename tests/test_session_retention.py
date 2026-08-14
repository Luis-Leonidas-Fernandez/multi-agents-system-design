from pathlib import Path


def test_enforce_recent_session_retention_keeps_root_files_and_latest_dirs(tmp_path):
    from features.web_scraping.application.session_retention import (
        enforce_recent_session_retention,
    )

    root = tmp_path / "data" / "sessions"
    root.mkdir(parents=True)
    (root / "sessions.db").write_text("db", encoding="utf-8")
    (root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    dirs = []
    for index, name in enumerate(["old", "mid", "new", "newest"]):
        session_dir = root / name
        session_dir.mkdir()
        (session_dir / "audit.json").write_text("{}", encoding="utf-8")
        timestamp = 1000 + index
        session_dir.touch()
        import os

        os.utime(session_dir, (timestamp, timestamp))
        dirs.append(session_dir)

    removed = enforce_recent_session_retention(root, keep=3)

    assert [path.name for path in removed] == ["old"]
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == [
        "mid",
        "new",
        "newest",
    ]
    assert (root / "sessions.db").is_file()
    assert (root / "events.jsonl").is_file()


def test_enforce_recent_session_retention_noops_for_missing_root(tmp_path):
    from features.web_scraping.application.session_retention import (
        enforce_recent_session_retention,
    )

    assert enforce_recent_session_retention(tmp_path / "missing", keep=3) == []
