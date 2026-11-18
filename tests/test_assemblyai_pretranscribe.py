"""scripts/assemblyai_pretranscribe.py: the transcript lands where a job
looks for it, under the id the app gives that link, and is never redone by
accident."""

import importlib.util
from pathlib import Path

import pytest

from llm.providers import keys

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "assemblyai_pretranscribe.py"
AAI_KEY = "aai0" + "c0ffee12" * 4


@pytest.fixture
def script(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("assemblyai_pretranscribe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def download(url, folder):
        out = folder / "audio.m4a"
        out.write_bytes(b"m4a")
        return out

    monkeypatch.setattr(module, "download_audio", download)
    monkeypatch.setattr(module.cloud, "_duration", lambda path: 7200.0)
    sent = []

    def transcribe(audio, video_id, transcripts, online, language=None):
        sent.append({"video_id": video_id, "online": online, "language": language})
        transcripts.mkdir(parents=True, exist_ok=True)
        (transcripts / f"{video_id}.json").write_text("{}", encoding="utf-8")
        return []

    monkeypatch.setattr(module.cloud, "transcribe", transcribe)
    module.sent = sent
    return module


def test_the_transcript_is_written_under_the_apps_id_for_the_link(script, tmp_path, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", AAI_KEY)
    terms = tmp_path / "terms.txt"
    terms.write_text("# a comment\nLos Santos\n\nanjir\n", encoding="utf-8")

    assert script.main(["https://www.youtube.com/watch?v=vfjrOsDrfRY", "--data-dir", str(tmp_path),
                        "--keyterms", str(terms)]) == 0

    (call,) = script.sent
    assert call["video_id"] == "vfjrOsDrfRY" and call["language"] == "id"
    assert call["online"] == {"backend": "assemblyai", "model": "universal-2", "data_dir": str(tmp_path),
                              "keyterms": ["Los Santos", "anjir"]}
    assert (tmp_path / "transcripts" / "vfjrOsDrfRY.json").exists()
    assert keys.load_key(tmp_path, "assemblyai") == AAI_KEY   # kept, encrypted, for next time


def test_an_existing_transcript_is_kept_unless_forced(script, tmp_path, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", AAI_KEY)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vfjrOsDrfRY.json").write_text("{}", encoding="utf-8")
    url = "https://youtu.be/vfjrOsDrfRY"

    script.main([url, "--data-dir", str(tmp_path)])
    assert script.sent == []
    script.main([url, "--data-dir", str(tmp_path), "--force", "--language", "auto"])
    assert len(script.sent) == 1 and script.sent[0]["language"] is None


def test_a_channel_page_is_refused(script, tmp_path):
    with pytest.raises(SystemExit):
        script.main(["https://www.twitch.tv/somechannel", "--data-dir", str(tmp_path)])
