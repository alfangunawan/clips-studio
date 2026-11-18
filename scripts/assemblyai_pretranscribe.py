"""Transcribe a stream with AssemblyAI for an installed Clips Kitty.

The installed app is a packaged build: its transcription choice can't be
changed without rebuilding it. But every job reads a cached transcript first
(transcription/transcriber.py: data/transcripts/<video id>.json) and skips
transcription when one is there. This writes that file, through exactly the
code a job would use (transcription/cloud.py), so the app then clips the video
with AssemblyAI's transcript instead of running Whisper on this PC.

    python scripts/assemblyai_pretranscribe.py <link> [--language id] [--keyterms file.txt]

Then paste the same link into Clips Kitty. Only the audio is downloaded here.

The key: the one saved for AssemblyAI in the app's data folder, else the
ASSEMBLYAI_API_KEY environment variable, else it is asked for (not shown as
you type). A key entered here is saved encrypted (Windows DPAPI) in the app's
data folder, like every other key the app keeps.
"""

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.providers import keys  # noqa: E402
from llm.providers.base import LLMError  # noqa: E402
from sources.dispatch import identify  # noqa: E402
from transcription import cloud  # noqa: E402

PRICE_PER_HOUR = {"universal-2": 0.15, "assemblyai-stt-3-7": 0.21}
KEYTERMS_PER_HOUR = 0.05


def default_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Clips Studio" / "data"


def read_keyterms(path: Path | None) -> list[str]:
    if not path:
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def api_key(data_dir: Path) -> str:
    key = keys.load_key(data_dir, "assemblyai")
    if key:
        return key
    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        key = getpass.getpass("AssemblyAI API key (not shown): ").strip()
    if not key:
        raise SystemExit("No AssemblyAI API key given.")
    keys.save_key(data_dir, "assemblyai", key)
    return key


def download_audio(url: str, folder: Path) -> Path:
    import yt_dlp

    options = {"format": "bestaudio/best", "outtmpl": str(folder / "audio.%(ext)s"),
               "quiet": True, "noprogress": False, "noplaylist": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    found = sorted(folder.glob("audio.*"))
    if not found:
        raise SystemExit("The audio didn't download.")
    return found[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="the YouTube, Twitch or Kick link you will paste into Clips Kitty")
    parser.add_argument("--language", default="id",
                        help="language code of the stream (default id); 'auto' lets AssemblyAI detect it")
    parser.add_argument("--model", default="universal-2", choices=sorted(PRICE_PER_HOUR))
    parser.add_argument("--keyterms", type=Path, help="a text file of names and slang, one per line")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="Clips Kitty's data folder (default: %%LOCALAPPDATA%%\\Clips Studio\\data)")
    parser.add_argument("--force", action="store_true", help="transcribe again even if a transcript exists")
    args = parser.parse_args(argv)

    source, video_id = identify(args.url)
    if not video_id or source == "local":
        raise SystemExit("That link doesn't point at one video. Use the link of the VOD or video itself.")
    transcripts = args.data_dir / "transcripts"
    target = transcripts / f"{video_id}.json"
    if target.exists() and not args.force:
        print(f"A transcript for {video_id} is already there: {target}\nUse --force to replace it.")
        return 0

    api_key(args.data_dir)  # saved in the encrypted store; cloud.transcribe reads it from there
    terms = read_keyterms(args.keyterms)
    language = None if args.language == "auto" else args.language
    online = {"backend": "assemblyai", "model": args.model, "data_dir": str(args.data_dir)}
    if terms:
        online["keyterms"] = terms

    with tempfile.TemporaryDirectory(prefix="ck-aai-") as tmp:
        print(f"Downloading the audio of {source} video {video_id}...")
        audio = download_audio(args.url, Path(tmp))
        hours = cloud._duration(audio) / 3600
        rate = PRICE_PER_HOUR[args.model] + (KEYTERMS_PER_HOUR if terms else 0.0)
        print(f"{hours:.2f} hours of audio: about ${hours * rate:.2f} on AssemblyAI "
              f"({args.model}{', with keyterms' if terms else ''}; parts with no speech aren't sent).")
        if target.exists():
            target.unlink()
        try:
            segments = cloud.transcribe(audio, video_id, transcripts, online, language=language)
        except LLMError as e:
            raise SystemExit(e.message) from e

    print(f"Done: {len(segments)} segments in {target}\n"
          "Now paste the same link into Clips Kitty; it will use this transcript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
