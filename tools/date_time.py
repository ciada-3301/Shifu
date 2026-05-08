from langchain_core.tools import tool
from datetime import datetime
import zoneinfo

@tool
def get_datetime(timezone: str = "Asia/Kolkata") -> str:
    """
    Returns the current date and time in multiple formats so the agent
    can pick whichever suits the task (scheduling, file naming, display, etc.).

    Args:
        timezone: IANA timezone string, e.g. "Asia/Kolkata", "UTC", "America/New_York".
                  Defaults to Asia/Kolkata (IST).

    Returns:
        A plain-text block with the current datetime in several formats.
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("Asia/Kolkata")

    now = datetime.now(tz)

    return (
        f"timezone   : {timezone}\n"
        f"iso8601    : {now.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"date       : {now.strftime('%Y-%m-%d')}\n"
        f"time       : {now.strftime('%H:%M:%S')}\n"
        f"human      : {now.strftime('%A, %d %B %Y %I:%M %p %Z')}\n"
        f"filename   : {now.strftime('%Y%m%d_%H%M%S')}\n"
        f"unix       : {int(now.timestamp())}\n"
    )