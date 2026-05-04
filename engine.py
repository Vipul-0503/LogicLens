"""
LogicLens Transcript Engine
============================
Extracts YouTube transcripts with timestamps using yt-dlp.
Supports cookie-based auth, VTT parsing, de-duplication, and clean structured output.
"""

import re
import os
import io
import tempfile
import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs

import yt_dlp

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LogicLens.Engine")


# Constants
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# VTT patterns
_VTT_TAG_RE = re.compile(r"<[^>]+>")                          # <c>, <i>, <b>, timestamp tags
_VTT_CUE_HEADER_RE = re.compile(                               # cue timing lines
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
)
_VTT_HEADER_RE = re.compile(r"^(WEBVTT|Kind:|Language:)")      # file-level headers
_TIMESTAMP_START_RE = re.compile(                              # captures HH:MM:SS from cue line
    r"^(\d{2}:\d{2}:\d{2})\.\d{3}\s+-->"
)
_WHITESPACE_RE = re.compile(r"\s+")


# TranscriptEntry helper (typed dict equivalent)
class TranscriptEntry:
    """A single transcript segment."""

    __slots__ = ("start", "text")

    def __init__(self, start: str, text: str) -> None:
        self.start = start   # "HH:MM:SS"
        self.text = text     # cleaned spoken words

    def to_dict(self) -> dict:
        return {"start": self.start, "text": self.text}

    def __repr__(self) -> str:
        return f"TranscriptEntry(start={self.start!r}, text={self.text!r})"


# Main Engine
class TranscriptEngine:
    """
    Extracts and structures YouTube transcripts using yt-dlp.

    Parameters
    ----------
    cookies_path : str | None
        Path to a Netscape-format cookies.txt file.  Pass ``None`` to skip.
    user_agent   : str
        HTTP User-Agent header sent with every request.
    """

    def __init__(
        self,
        cookies_path: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cookies_path = cookies_path
        self.user_agent = user_agent
        self._validate_cookies()

    # Public API
    def extract(self, url: str) -> list[dict]:
        """
        Full pipeline: URL → structured transcript list.

        Returns
        -------
        list[dict]
            Each element: ``{"start": "HH:MM:SS", "text": "..."}``
        Raises
        ------
        ValueError
            If the video ID cannot be parsed from *url*.
        RuntimeError
            If no English subtitle track is available.
        """
        video_id = self._extract_video_id(url)
        logger.info("Processing video ID: %s", video_id)

        vtt_content = self._fetch_vtt(video_id)
        entries = self._parse_vtt(vtt_content)
        entries = self._deduplicate(entries)

        logger.info("Extracted %d transcript segments.", len(entries))
        return [e.to_dict() for e in entries]

    # Step 1 – Extract Video ID
    @staticmethod
    def _extract_video_id(url: str) -> str:
        """
        Parses the YouTube video ID from various URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
        """
        parsed = urlparse(url)

        # youtu.be/<id>
        if parsed.netloc in ("youtu.be",):
            vid = parsed.path.lstrip("/").split("/")[0]
            if vid:
                return vid

        # youtube.com/shorts/<id>
        if "/shorts/" in parsed.path:
            vid = parsed.path.split("/shorts/")[-1].split("/")[0]
            if vid:
                return vid

        # Standard ?v= param
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]

        raise ValueError(f"Cannot extract video ID from URL: {url!r}")

    # Step 2 – Fetch VTT via yt-dlp
    def _build_ydl_opts(self, subtitle_buffer: dict) -> dict:
        """Construct yt-dlp options dict."""
        opts: dict = {
            # Skip audio/video download entirely
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",

            # Write subtitle file into a temp dir we control
            "outtmpl": subtitle_buffer["outtmpl"],

            # Bot-detection mitigation
            "http_headers": {"User-Agent": self.user_agent},

            # Suppress yt-dlp console noise; we use our own logger
            "quiet": True,
            "no_warnings": False,
            "logger": _YtDlpLogger(),

            # Retry / timeout
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 30,
        }

        if self.cookies_path:
            opts["cookiefile"] = self.cookies_path

        return opts

    def _fetch_vtt(self, video_id: str) -> str:
        """
        Uses yt-dlp to download the subtitle file to a temp directory,
        reads it, and returns the raw VTT string.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "%(id)s")
            buf = {"outtmpl": outtmpl}
            opts = self._build_ydl_opts(buf)
            url = f"https://www.youtube.com/watch?v={video_id}"

            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as exc:
                raise RuntimeError(f"yt-dlp download error: {exc}") from exc

            # yt-dlp writes e.g. <video_id>.en.vtt or <video_id>.en-orig.vtt
            vtt_file = self._find_vtt_file(tmpdir, video_id)
            if vtt_file is None:
                raise RuntimeError(
                    "No English subtitle file was produced. "
                    "The video may not have English subtitles."
                )

            with open(vtt_file, "r", encoding="utf-8") as fh:
                return fh.read()

    @staticmethod
    def _find_vtt_file(directory: str, video_id: str) -> Optional[str]:
        """Return the first .vtt file in *directory* matching the video id."""
        for fname in os.listdir(directory):
            if fname.startswith(video_id) and fname.endswith(".vtt"):
                return os.path.join(directory, fname)
        # Fallback: any .vtt
        for fname in os.listdir(directory):
            if fname.endswith(".vtt"):
                return os.path.join(directory, fname)
        return None

    # Step 3 – Parse VTT → List[TranscriptEntry]
    @classmethod
    def _parse_vtt(cls, vtt: str) -> list[TranscriptEntry]:
        """
        Parses raw WebVTT text into a list of TranscriptEntry objects.

        YouTube VTT looks like:

            WEBVTT
            Kind: captions

            00:00:01.280 --> 00:00:03.440
            <00:00:01.280><c> Hello</c><00:00:02.120><c> world</c>

        We capture the start timestamp from the cue header and the
        cleaned text from the payload lines.
        """
        entries: list[TranscriptEntry] = []
        lines = vtt.splitlines()

        current_start: Optional[str] = None
        text_lines: list[str] = []

        for line in lines:
            line = line.strip()

            # Skip file-level headers and blank lines between blocks
            if not line or _VTT_HEADER_RE.match(line):
                if current_start and text_lines:
                    cleaned = cls._clean_text(" ".join(text_lines))
                    if cleaned:
                        entries.append(TranscriptEntry(current_start, cleaned))
                    current_start = None
                    text_lines = []
                continue

            # Cue timing line: "00:00:01.280 --> 00:00:03.440 ..."
            m = _TIMESTAMP_START_RE.match(line)
            if m:
                # Flush previous cue if any
                if current_start and text_lines:
                    cleaned = cls._clean_text(" ".join(text_lines))
                    if cleaned:
                        entries.append(TranscriptEntry(current_start, cleaned))
                current_start = m.group(1)
                text_lines = []
                continue

            # Cue numeric identifier lines (e.g., "1", "2") — skip
            if line.isdigit():
                continue

            # Text payload
            if current_start is not None:
                text_lines.append(line)

        # Flush last cue
        if current_start and text_lines:
            cleaned = cls._clean_text(" ".join(text_lines))
            if cleaned:
                entries.append(TranscriptEntry(current_start, cleaned))

        return entries

    # Step 4 – Clean Text
    @staticmethod
    def _clean_text(raw: str) -> str:
        """
        Strips all VTT inline tags and normalises whitespace.

        Input example:
            "<00:00:01.280><c> Hello</c> <00:00:02.120><c> world</c>"
        Output:
            "Hello world"
        """
        text = _VTT_TAG_RE.sub("", raw)          # remove all <...> tags
        text = re.sub(r">>\s*", "", text)
        text = _WHITESPACE_RE.sub(" ", text)      # collapse whitespace
        return text.strip()

    # Step 5 – De-duplicate
    @staticmethod
    def _deduplicate(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """
        Removes consecutive duplicate or overlapping segments.

        YouTube auto-captions often repeat the same phrase across adjacent
        cue blocks (rolling-window style). We keep only the first occurrence
        of any text that is a substring of the immediately following entry,
        and skip exact duplicates entirely.
        """
        if not entries:
            return entries

        deduped: list[TranscriptEntry] = [entries[0]]

        for entry in entries[1:]:
            prev = deduped[-1]
            # Skip exact duplicates
            if entry.text == prev.text:
                continue
            # Skip if this entry's text is fully contained in the previous one
            if entry.text in prev.text:
                continue
            # Skip if the previous text is fully contained in this entry
            # (rolling-window expansion — replace the previous with this one)
            if prev.text in entry.text:
                deduped[-1] = entry
                continue
            deduped.append(entry)

        return deduped

    # Validation helpers
    def _validate_cookies(self) -> None:
        if self.cookies_path is None:
            return
        if not os.path.isfile(self.cookies_path):
            raise FileNotFoundError(
                f"cookies.txt not found at: {self.cookies_path!r}"
            )
        logger.info("Using cookies file: %s", self.cookies_path)


# yt-dlp logger shim (routes yt-dlp messages into our logger)
class _YtDlpLogger:
    def debug(self, msg: str) -> None:
        if msg.startswith("[debug]"):
            logger.debug(msg)

    def info(self, msg: str) -> None:
        logger.info(msg)

    def warning(self, msg: str) -> None:
        logger.warning(msg)

    def error(self, msg: str) -> None:
        logger.error(msg)


# CLI entry-point (quick smoke-test)
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python engine.py <youtube_url> [cookies.txt]")
        sys.exit(1)

    youtube_url = sys.argv[1]
    cookies = sys.argv[2] if len(sys.argv) > 2 else None

    engine = TranscriptEngine(cookies_path=cookies)

    try:
        transcript = engine.extract(youtube_url)
        print(json.dumps(transcript, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)