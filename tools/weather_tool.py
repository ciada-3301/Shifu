"""
tools/get_weather.py — Shifu's Weather Tool (Open-Meteo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Free, no API key, no registration. Powered by Open-Meteo.

Supports three modes determined by the optional `date` parameter:
  • No date / "today"   → current conditions + 3-day forecast  (forecast API)
  • Past date           → historical daily summary             (archive API)
  • Future date         → daily forecast for that specific day (forecast API)

Date strings accepted: "today", "yesterday", "YYYY-MM-DD"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import urllib.request
import urllib.parse
import json
from datetime import date, timedelta
from langchain_core.tools import tool

# WMO weather interpretation codes → human description
_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}


def _geocode(location: str) -> tuple[float, float, str]:
    """Resolve a city/place name to (lat, lon, display_name) via Open-Meteo geocoding."""
    params = urllib.parse.urlencode({
        "name": location, "count": 1, "language": "en", "format": "json"
    })
    url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
    with urllib.request.urlopen(url, timeout=8) as r:
        data = json.loads(r.read())
    if not data.get("results"):
        raise ValueError(f"Location not found: {location!r}")
    r = data["results"][0]
    name = f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".strip(", ")
    return r["latitude"], r["longitude"], name


def _resolve_date(date_str: str | None) -> date | None:
    """
    Parse the user-supplied date string into a date object.
    Returns None when the user wants current conditions (no date / "today").
    """
    if not date_str or date_str.strip().lower() in ("", "today", "now", "current"):
        return None                        # → current conditions + 3-day forecast
    s = date_str.strip().lower()
    if s == "yesterday":
        return date.today() - timedelta(days=1)
    if s == "tomorrow":
        return date.today() + timedelta(days=1)
    try:
        return date.fromisoformat(s)       # expects "YYYY-MM-DD"
    except ValueError:
        raise ValueError(
            f"Unrecognised date {date_str!r}. "
            "Use 'today', 'yesterday', 'tomorrow', or 'YYYY-MM-DD'."
        )


def _fetch_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read())


# ── Mode A: current conditions + 3-day forecast ───────────────────────────────
def _weather_current(lat: float, lon: float, display: str) -> str:
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "current_weather": "true",
        "hourly": "relativehumidity_2m,apparent_temperature",
        "daily": (
            "weathercode,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,windspeed_10m_max,sunrise,sunset"
        ),
        "timezone": "auto",
        "forecast_days": 4,
    })
    data = _fetch_url(f"https://api.open-meteo.com/v1/forecast?{params}")

    cw   = data["current_weather"]
    temp = cw["temperature"]
    wind = cw["windspeed"]
    wdir = cw["winddirection"]
    desc = _WMO.get(cw["weathercode"], f"code {cw['weathercode']}")
    tz   = data.get("timezone", "")

    cur_hour_idx = 0
    try:
        cur_time = cw["time"][:13]
        hours = data["hourly"]["time"]
        cur_hour_idx = next((i for i, t in enumerate(hours) if t.startswith(cur_time)), 0)
    except Exception:
        pass
    humidity   = data["hourly"]["relativehumidity_2m"][cur_hour_idx]
    feels_like = data["hourly"]["apparent_temperature"][cur_hour_idx]

    lines = [
        f"📍 {display}  ({lat:.2f}, {lon:.2f})  [{tz}]",
        "",
        "NOW",
        f"  condition  : {desc}",
        f"  temperature: {temp}°C  (feels like {feels_like}°C)",
        f"  humidity   : {humidity}%",
        f"  wind       : {wind} km/h @ {wdir}°",
        "",
        "3-DAY FORECAST",
    ]
    daily = data["daily"]
    for i in range(1, 4):
        try:
            d_      = daily["time"][i]
            hi      = daily["temperature_2m_max"][i]
            lo      = daily["temperature_2m_min"][i]
            precip  = daily["precipitation_sum"][i]
            wmax    = daily["windspeed_10m_max"][i]
            sunrise = daily["sunrise"][i][11:]
            sunset  = daily["sunset"][i][11:]
            cond    = _WMO.get(daily["weathercode"][i], "?")
            lines.append(
                f"  {d_}  {cond:<24} {lo}°C – {hi}°C  "
                f"rain {precip}mm  wind {wmax}km/h  "
                f"↑{sunrise} ↓{sunset}"
            )
        except Exception:
            pass

    return "\n".join(lines)


# ── Mode B: single-day historical or specific future date ─────────────────────
def _weather_for_date(lat: float, lon: float, display: str, target: date) -> str:
    today = date.today()
    is_past = target < today

    if is_past:
        # Open-Meteo archive API — free, covers from 1940-01-01 to ~5 days ago
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Forecast API supports up to 16 days ahead
        base_url = "https://api.open-meteo.com/v1/forecast"

    date_str = target.isoformat()
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": date_str,
        "end_date":   date_str,
        "daily": (
            "weathercode,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,windspeed_10m_max,"
            "windgusts_10m_max,sunrise,sunset,"
            "precipitation_hours,shortwave_radiation_sum"
        ),
        "hourly": "temperature_2m,relativehumidity_2m,apparent_temperature,precipitation",
        "timezone": "auto",
    })
    data = _fetch_url(f"{base_url}?{params}")

    tz     = data.get("timezone", "")
    daily  = data["daily"]
    hourly = data["hourly"]

    # Daily summary
    hi      = daily["temperature_2m_max"][0]
    lo      = daily["temperature_2m_min"][0]
    precip  = daily["precipitation_sum"][0]
    wmax    = daily["windspeed_10m_max"][0]
    gusts   = daily.get("windgusts_10m_max", [None])[0]
    sunrise = daily["sunrise"][0][11:]   # "HH:MM"
    sunset  = daily["sunset"][0][11:]
    cond    = _WMO.get(daily["weathercode"][0], f"code {daily['weathercode'][0]}")
    rain_h  = daily.get("precipitation_hours", [None])[0]
    rad     = daily.get("shortwave_radiation_sum", [None])[0]

    label = "HISTORICAL" if is_past else "FORECAST"
    lines = [
        f"📍 {display}  ({lat:.2f}, {lon:.2f})  [{tz}]",
        "",
        f"{label} — {date_str}",
        f"  condition  : {cond}",
        f"  temp range : {lo}°C – {hi}°C",
        f"  precipitation: {precip} mm" + (f"  ({rain_h}h of rain)" if rain_h else ""),
        f"  wind max   : {wmax} km/h" + (f"  (gusts {gusts} km/h)" if gusts else ""),
        f"  sunrise / sunset: {sunrise} / {sunset}",
    ]
    if rad is not None:
        lines.append(f"  solar radiation: {rad} MJ/m²")

    # Hourly breakdown (temperature + humidity + precipitation)
    lines += ["", "HOURLY BREAKDOWN"]
    times   = hourly["time"]
    temps   = hourly["temperature_2m"]
    hums    = hourly["relativehumidity_2m"]
    feels   = hourly["apparent_temperature"]
    rains   = hourly["precipitation"]

    for i, t in enumerate(times):
        hh = t[11:]  # "HH:MM"
        lines.append(
            f"  {hh}  {temps[i]:5.1f}°C (feels {feels[i]:.1f}°C)"
            f"  humidity {hums[i]}%"
            f"  rain {rains[i]:.1f}mm"
        )

    return "\n".join(lines)


# ── Public tool ───────────────────────────────────────────────────────────────
@tool
def get_weather(location: str = "Kolkata", date: str = "") -> str:
    """
    Get weather data for any location and date. No API key required.

    Three modes depending on the `date` argument:
      • Omitted / "today"   → current conditions + 3-day forecast
      • Past date           → historical daily summary + hourly breakdown
      • Specific future date → daily forecast for that exact day

    Args:
        location: City or place name, e.g. "Kolkata", "London", "New York".
                  Defaults to Kolkata.
        date:     One of:
                    ""           — current conditions + 3-day forecast (default)
                    "today"      — same as ""
                    "yesterday"  — previous calendar day (historical archive)
                    "tomorrow"   — next calendar day (forecast)
                    "YYYY-MM-DD" — any specific date past or future

    Returns:
        A plain-text weather report. Past dates include hourly breakdown.
    """
    try:
        lat, lon, display = _geocode(location)
    except Exception as e:
        return f"❌ Geocoding failed: {e}"

    try:
        target = _resolve_date(date)
    except ValueError as e:
        return f"❌ Date error: {e}"

    try:
        if target is None:
            return _weather_current(lat, lon, display)
        else:
            return _weather_for_date(lat, lon, display, target)
    except Exception as e:
        return f"❌ Weather fetch failed: {e}"