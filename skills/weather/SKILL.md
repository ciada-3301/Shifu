\---
name: weather
description: "Use this skill whenever the user asks about current weather, temperature, humidity, wind, rain, forecast, or whether to bring an umbrella/jacket for any location. Also handles HISTORICAL weather ('what was the weather yesterday', 'weather on 2026-05-01') and SPECIFIC FUTURE dates ('forecast for next Monday'). Triggers: 'what's the weather in X', 'will it rain tomorrow', 'how hot is it in X', 'weather yesterday in X', 'weather on YYYY-MM-DD', 'save weather to excel', 'weather forecast for X'. Calls get_weather which uses Open-Meteo — free, no API key, works for any city worldwide. Default location is Kolkata if none specified. DO NOT use the browser tool or web search for weather — this tool covers current, historical, and forecast data natively."
compatibility: "Shifu / LangGraph agent — requires get_weather tool registered in the agent's toolset (tools/get_weather.py)"
---

# Weather Skill

## What this skill does

`get_weather` fetches live weather data from [Open-Meteo](https://open-meteo.com/) — a
free, no-registration, no-API-key service. It operates in three modes:

| Mode | When | API used |
|---|---|---|
| **Current + 3-day forecast** | No date / `"today"` | `api.open-meteo.com/v1/forecast` |
| **Historical daily + hourly** | Any past date, `"yesterday"` | `archive-api.open-meteo.com/v1/archive` |
| **Specific future day** | `"tomorrow"` or `"YYYY-MM-DD"` | `api.open-meteo.com/v1/forecast` |

> ⚠️ **Never route weather queries to the browser tool or web search.**
> The archive API covers historical data back to 1940. The forecast API covers up to
> 16 days ahead. There is no weather query that requires scraping a webpage.

---

## Tool signature

```python
get_weather(location: str = "Kolkata", date: str = "") -> str
```

| Parameter  | Type  | Default      | Accepted values |
|------------|-------|--------------|-----------------|
| `location` | `str` | `"Kolkata"`  | Any city or place name. For ambiguous names add country/state: `"Springfield, Illinois"` |
| `date`     | `str` | `""` (today) | `""`, `"today"`, `"yesterday"`, `"tomorrow"`, `"YYYY-MM-DD"` |

---

## When to call `get_weather`

| User says… | `date=` to pass |
|---|---|
| "What's the weather in Mumbai?" | `""` (omit) |
| "What was the weather yesterday in Delhi?" | `"yesterday"` |
| "Weather on May 1st 2026 in London" | `"2026-05-01"` |
| "Will it rain tomorrow in Chennai?" | `"tomorrow"` |
| "Forecast for next Thursday" (today is Thu May 7) | `"2026-05-14"` |
| "Save yesterday, today and tomorrow's weather to Excel" | Three calls: `"yesterday"`, `""`, `"tomorrow"` |

### Do NOT call `get_weather` for:
- **Climate questions** ("what is the average rainfall in Kerala") — answer from knowledge
- **Historical events** ("weather during the 1999 Odisha cyclone") — answer from knowledge

---

## Output formats

### Current conditions + 3-day forecast (date omitted or "today")
```
📍 Kolkata, West Bengal, India  (22.57, 88.36)  [Asia/Kolkata]

NOW
  condition  : Partly cloudy
  temperature: 34°C  (feels like 40°C)
  humidity   : 72%
  wind       : 18 km/h @ 210°

3-DAY FORECAST
  2026-05-08  Showers                  27°C – 36°C  rain 4.2mm  wind 22km/h  ↑05:13 ↓18:27
  2026-05-09  Light rain               26°C – 34°C  rain 8.1mm  wind 19km/h  ↑05:13 ↓18:27
  2026-05-10  Partly cloudy            25°C – 35°C  rain 0.0mm  wind 15km/h  ↑05:12 ↓18:28
```

### Historical or specific-date (any date value supplied)
```
📍 Kolkata, West Bengal, India  (22.57, 88.36)  [Asia/Kolkata]

HISTORICAL — 2026-05-06
  condition  : Light rain
  temp range : 27°C – 35°C
  precipitation: 6.2 mm  (3h of rain)
  wind max   : 24 km/h  (gusts 38 km/h)
  sunrise / sunset: 05:13 / 18:26
  solar radiation: 14.3 MJ/m²

HOURLY BREAKDOWN
  00:00   28.4°C (feels 33.1°C)  humidity 81%  rain 0.0mm
  01:00   28.1°C (feels 32.8°C)  humidity 82%  rain 0.0mm
  ...
  14:00   34.2°C (feels 40.0°C)  humidity 64%  rain 2.1mm
  ...
```

All temperatures: **°C**. Wind: **km/h**. Precipitation: **mm**.
Times are **local** to the queried location (`timezone=auto`).

---

## Multi-date + Excel workflow

For requests like *"get yesterday, today, and tomorrow's weather and save to Excel"*:

```
Step 1 — Call get_weather three times:
    yesterday_data = get_weather({"location": "Kolkata", "date": "yesterday"})
    today_data     = get_weather({"location": "Kolkata", "date": ""})
    tomorrow_data  = get_weather({"location": "Kolkata", "date": "tomorrow"})

Step 2 — Parse each result into a row/table using the structured fields above.

Step 3 — Pass the combined data to the xlsx tool to write the Excel file.
         Do NOT use the browser tool. Do NOT do web searches.
```

The xlsx tool should receive a structured dict or list of dicts, not the raw
text output. Extract the relevant fields (date, condition, high, low, precipitation,
wind, humidity) from each result before passing to xlsx.

**Suggested Excel schema:**

| Date | Type | Condition | Temp Min (°C) | Temp Max (°C) | Precipitation (mm) | Wind Max (km/h) | Humidity (%) |
|------|------|-----------|---------------|---------------|-------------------|-----------------|--------------|

---

## Error cases

| Error string | Cause | Action |
|---|---|---|
| `❌ Geocoding failed: Location not found: …` | Place name unrecognised | Ask user to clarify or use a nearby city |
| `❌ Date error: Unrecognised date …` | Bad date format | Correct to `YYYY-MM-DD` or keyword |
| `❌ Weather fetch failed: …` | Network or API error | Inform user, suggest retry |

**Never hallucinate weather data** when the tool returns `❌`.

---

## Example reasoning traces

**"Get yesterday, today and tomorrow's weather and save to Excel"**
```
Thought: Three date-specific weather calls, then xlsx. No browser needed.
Action: get_weather({"location": "Kolkata", "date": "yesterday"})
Action: get_weather({"location": "Kolkata", "date": ""})
Action: get_weather({"location": "Kolkata", "date": "tomorrow"})
Thought: Parse results → build rows → pass to xlsx tool.
Action: xlsx_write({...})
```

**"What was the weather in Mumbai on 1st May?"**
```
Thought: Specific past date. Use archive mode.
Action: get_weather({"location": "Mumbai", "date": "2026-05-01"})
```

**"Will it rain in Bangalore tomorrow?"**
```
Action: get_weather({"location": "Bangalore", "date": "tomorrow"})
Observation: precipitation: 8.4 mm
Response: Yes, expect about 8mm of rain in Bangalore tomorrow.
```

---

## Dependencies & setup

```python
# Pure stdlib only — no extra pip installs.
# Tool file: tools/get_weather.py
# Register in your agent:

from tools.get_weather import get_weather
tools = [..., get_weather]
```

Open-Meteo forecast API: https://open-meteo.com/en/docs  
Open-Meteo archive API:  https://open-meteo.com/en/docs/historical-weather-api  
Geocoding API:           https://open-meteo.com/en/docs/geocoding-api