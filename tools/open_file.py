"""
tools/open_file.py — Shifu's File & Application Launcher Tool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Opens files, plays audio/video, launches applications, and handles
URLs — all via the OS's default handlers (or an explicit app).

Supported actions:
  • open      — open any file with its default application
  • play      — alias for open, optimised label for media files
  • launch    — launch a named application (optionally with args)
  • reveal    — reveal a file in the system file manager / Finder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ── Internal helpers ──────────────────────────────────────────────────────────

_SYSTEM = platform.system()  # "Darwin" | "Linux" | "Windows"

_AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"}
_DOC_EXT   = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
              ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"}

PLAYGROUND_DIR = Path("")


def _resolve_path(path: str) -> Path:
    """
    Resolve a relative path against Playground/, stripping any redundant
    leading 'Playground/' prefix first to avoid double-nesting.
    """
    p = Path(path)
    if not p.is_absolute():
        try:
            p = p.relative_to("")
        except ValueError:
            pass  # doesn't start with Playground/, use as-is
        p = PLAYGROUND_DIR / p
    return p.expanduser().resolve()


def _os_open(path_or_url: str) -> tuple[bool, str]:
    """Open *path_or_url* with the OS default handler."""
    if _SYSTEM == "Darwin":
        cmd = ["open", path_or_url]
    elif _SYSTEM == "Windows":
        cmd = ["cmd", "/c", "start", "", path_or_url]
    else:
        for opener in ("xdg-open", "gio open", "mimeopen"):
            bin_name = opener.split()[0]
            if _which(bin_name):
                cmd = shlex.split(opener) + [path_or_url]
                break
        else:
            return False, "No suitable file-opener found (xdg-open / gio / mimeopen)."

    return _run(cmd)


def _os_reveal(path: str) -> tuple[bool, str]:
    """Reveal *path* in the system file manager."""
    p = Path(path).resolve()
    if _SYSTEM == "Darwin":
        cmd = ["open", "-R", str(p)]
    elif _SYSTEM == "Windows":
        cmd = ["explorer", "/select,", str(p)]
    else:
        cmd = ["xdg-open", str(p.parent)]
    return _run(cmd)


def _launch_app(app: str, args: list[str]) -> tuple[bool, str]:
    """Launch *app* (name or full path) with optional *args*."""
    if _SYSTEM == "Darwin":
        if not app.endswith(".app") and not os.sep in app:
            ok, msg = _run(["open", "-a", app] + args)
            if ok:
                return ok, msg
        cmd = [app] + args
    elif _SYSTEM == "Windows":
        cmd = [app] + args
    else:
        cmd = shlex.split(app) + args if " " in app else [app] + args

    return _run(cmd)


def _run(cmd: list[str], timeout: int = 8) -> tuple[bool, str]:
    """Run *cmd* in a detached subprocess; return (success, message)."""
    try:
        result = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(0.4)
        poll = result.poll()
        if poll is not None and poll != 0:
            err = result.stderr.read().decode(errors="replace").strip()
            return False, f"Process exited with code {poll}: {err or '(no stderr)'}"
        return True, f"Launched: {' '.join(str(c) for c in cmd)}"
    except FileNotFoundError:
        return False, f"Executable not found: {cmd[0]!r}"
    except Exception as exc:
        return False, f"Error launching {cmd[0]!r}: {exc}"


def _which(name: str) -> Optional[str]:
    import shutil
    return shutil.which(name)


def _classify(path: str) -> str:
    """Return a human-readable category for the path extension."""
    ext = Path(path).suffix.lower()
    if ext in _AUDIO_EXT:
        return "audio file"
    if ext in _VIDEO_EXT:
        return "video file"
    if ext in _IMAGE_EXT:
        return "image file"
    if ext in _DOC_EXT:
        return "document"
    if path.startswith(("http://", "https://", "ftp://")):
        return "URL"
    return "file"


# ── LangChain Tool ────────────────────────────────────────────────────────────

@tool
def open_file(
    path: str,
    action: str = "open",
    app: str = "",
    app_args: str = "",
) -> str:
    """Open, play, or launch files, media, applications, and URLs on the host OS.

    Args:
        path:     Absolute or relative path to a file/directory/URL, OR the
                  name / path of an application when action='launch'.
                  Examples:
                    "Playground/report.pdf"
                    "/home/user/Music/song.mp3"
                    "https://example.com"
                    "firefox"   (when action='launch')

        action:   What to do. One of:
                    "open"   – open with the OS default app (default)
                    "play"   – alias of "open", semantic sugar for media
                    "launch" – launch a named application (path = app name/path)
                    "reveal" – show the file in Finder / File Manager / Nautilus

        app:      (Optional) Force a specific application to open the file,
                  e.g. app="vlc" or app="code".  Ignored for action='launch'.

        app_args: (Optional) Space-separated extra arguments forwarded to the
                  application, e.g. app_args="--fullscreen".
                  For action='launch', these are passed as CLI arguments.

    Returns:
        A plain-English status string describing what happened.
    """
    action = action.strip().lower()
    extra_args = shlex.split(app_args) if app_args.strip() else []

    valid_actions = ("open", "play", "launch", "reveal")
    if action not in valid_actions:
        return (
            f"❌ Unknown action '{action}'. "
            f"Choose one of: {', '.join(valid_actions)}."
        )

    # ── LAUNCH ────────────────────────────────────────────────────────────
    if action == "launch":
        if not path:
            return "❌ Provide the application name or path in the 'path' argument."
        ok, msg = _launch_app(path, extra_args)
        if ok:
            return f"✅ Application launched: {path!r}"
        return f"❌ Failed to launch {path!r}: {msg}"

    # ── Resolve file path ─────────────────────────────────────────────────
    is_url = path.startswith(("http://", "https://", "ftp://"))
    if not is_url:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return (
                f"❌ File not found: {path!r}\n"
                f"   Resolved to: {resolved}"
            )
        target = str(resolved)
        category = _classify(target)
    else:
        target = path
        category = "URL"

    # ── REVEAL ────────────────────────────────────────────────────────────
    if action == "reveal":
        ok, msg = _os_reveal(target)
        if ok:
            return f"✅ Revealed {category} in file manager: {target!r}"
        return f"❌ Could not reveal file: {msg}"

    # ── OPEN / PLAY ───────────────────────────────────────────────────────
    if app.strip():
        app_name = app.strip()
        if _SYSTEM == "Darwin":
            cmd_args = ["open", "-a", app_name, target] + extra_args
        else:
            if not _which(app_name):
                return f"❌ Application not found: {app_name!r}"
            cmd_args = [app_name, target] + extra_args
        ok, msg = _run(cmd_args)
        verb = "Playing" if action == "play" else "Opening"
        if ok:
            return f"✅ {verb} {category} with {app_name!r}: {target!r}"
        return f"❌ Failed to open with {app_name!r}: {msg}"

    ok, msg = _os_open(target)
    verb = "Playing" if action == "play" else "Opening"
    if ok:
        return f"✅ {verb} {category}: {target!r}"
    return f"❌ Could not open {category}: {msg}"