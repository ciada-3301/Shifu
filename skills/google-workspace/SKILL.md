---
name: google-workspace
description: "Load this skill before using gmeet_create_meeting, gdrive_list_files, gdrive_upload_file, gdrive_download_file, gdrive_share_file, gdrive_delete_file, gdrive_create_folder, gcalendar_list_events, gcalendar_create_event, gcalendar_update_event, gcalendar_delete_event, gmeet_get_link_for_event, gmaps_search_places, gmaps_get_directions, gmaps_geocode, gmaps_place_details. Triggers: any mission involving Google Meet, Google Drive, Google Calendar, Google Maps, scheduling, meetings, file sharing, directions, or place search."
tags: "google, drive, calendar, meet, maps, scheduling, file-sharing, places, directions, gmeet, gcalendar, gdrive, gmaps"
---

# Google Workspace Skill for Shifu

## CRITICAL — Read before writing any plan

The ONLY tools available for Google services are the ones listed below.
**Do not invent tool names. Do not call methods that are not in this list.**
If a user asks for something not covered by these tools, say so clearly.

### Complete tool list

| Tool name (exact) | What it does |
|---|---|
| `gmeet_create_meeting` | Create a Google Meet + Calendar event, returns the join link |
| `gmeet_get_link_for_event` | Get the Meet link for an existing Calendar event |
| `gdrive_list_files` | Search / list files in Google Drive |
| `gdrive_upload_file` | Upload a local file from `Playground/` to Drive |
| `gdrive_download_file` | Download a Drive file into `Playground/` |
| `gdrive_create_folder` | Create a folder in Drive |
| `gdrive_share_file` | Share a Drive file with an email address |
| `gdrive_delete_file` | Permanently delete a Drive file |
| `gcalendar_list_events` | List upcoming Calendar events |
| `gcalendar_create_event` | Create a Calendar event (no Meet link) |
| `gcalendar_update_event` | Update an existing Calendar event |
| `gcalendar_delete_event` | Delete a Calendar event |
| `gmaps_search_places` | Text search for places (restaurants, hospitals, etc.) |
| `gmaps_get_directions` | Turn-by-turn directions between two points |
| `gmaps_geocode` | Convert an address to lat/lng |
| `gmaps_place_details` | Full details (hours, phone, website) for a place |

**Gmail / email is NOT available.** If the user asks to read, send, or summarise
emails, respond: "I don't have a Gmail tool connected yet. I can only access
Drive, Calendar, Meet, and Maps."

---

## Auth & environment prerequisites

| Item | What it is | Where it lives |
|---|---|---|
| `GOOGLE_CREDENTIALS_FILE` | OAuth client JSON from Google Cloud Console | Project root |
| `GOOGLE_TOKEN_FILE` | Auto-generated pickle after first login | Project root (auto-created) |
| `GOOGLE_MAPS_API_KEY` | API key for Places / Directions / Geocoding | `.env` |

First-run: a browser window opens for OAuth consent. After approval,
`google_token.pickle` is written and all future calls are silent.

---

## Google Meet — most common request

**User says:** "create a meeting", "schedule a Google Meet", "make a Meet link"

**Correct plan:**
```
1. load_skill(google-workspace)      <- always step 1
2. gmeet_create_meeting(...)         <- one tool call, returns the link
3. DONE summary with meet_link
```

**Never do this:**
- Do NOT call `google_meet.create_meeting()` — that tool does not exist
- Do NOT call `authenticate()` — auth is handled automatically inside the tool
- Do NOT ask the user for a title/time if they didn't provide one — use
  sensible defaults (title: "Quick Meeting", duration: 1 hour from now)

### gmeet_create_meeting — argument reference

```python
gmeet_create_meeting(
    title          = "Team Sync",                    # meeting name
    start_datetime = "2025-08-10T15:00:00+05:30",   # ISO 8601, IST = +05:30
    end_datetime   = "2025-08-10T16:00:00+05:30",
    attendee_emails= "alice@gmail.com, bob@gmail.com",  # comma-separated, or ""
    description    = "Weekly catch-up",              # optional
    calendar_id    = "primary",                      # always "primary"
)
```

**If the user gives no time:** default to 1 hour from the current time, IST.
**If the user gives no title:** use "Quick Meeting".
**If the user gives no attendees:** pass `attendee_emails=""` — that is fine,
the Meet link still works and the user can share it manually.

### Reading the response

The tool returns JSON. Extract these fields:

```json
{
  "meet_link":     "https://meet.google.com/xxx-yyyy-zzz",
  "calendar_link": "https://calendar.google.com/...",
  "event_id":      "abc123",
  "title":         "Team Sync"
}
```

Always include `meet_link` verbatim in the final summary.

---

## Google Calendar

### Datetime format — IST mandatory

All datetime arguments must be ISO 8601 with `+05:30` offset:

```
2025-08-10T15:30:00+05:30
```

Never use bare dates (`"2025-08-10"`) for timed events.
If the user says "3pm tomorrow", compute the correct ISO string with `+05:30`.

### gcalendar_list_events defaults

```python
gcalendar_list_events(calendar_id="primary", days_ahead=7, max_results=20)
```

Present results as a numbered list, not raw JSON.

---

## Google Drive

### Finding files before acting on them

Always call `gdrive_list_files` first to get the `id` before calling
`gdrive_download_file`, `gdrive_share_file`, or `gdrive_delete_file`.
Never guess or construct a file ID.

### Drive query syntax

| Goal | Query string |
|---|---|
| Find by name | `name contains 'report'` |
| Exact name | `name = 'Budget.xlsx'` |
| By MIME type | `mimeType = 'application/pdf'` |
| Inside a folder | `'<folder_id>' in parents` |

### Common workflow: upload + share

```
1. gdrive_upload_file(local_path="Playground/report.pdf")  -> parse id
2. gdrive_share_file(file_id=<id>, email="...", role="reader")
3. DONE: include webViewLink
```

---

## Google Maps

Maps tools require only `GOOGLE_MAPS_API_KEY` — no OAuth needed.

### Location bias for India

Pass `location="lat,lng"` to `gmaps_search_places` when the user implies a city:

| City | lat,lng |
|---|---|
| Patna | 25.5941,85.1376 |
| Delhi | 28.6139,77.2090 |
| Mumbai | 19.0760,72.8777 |
| Bengaluru | 12.9716,77.5946 |

### Directions + place search chain

```
1. gmaps_search_places(query="...", location="25.5941,85.1376")  -> pick address
2. gmaps_get_directions(origin="user location", destination=<address>, mode="driving")
3. DONE: distance, duration, top steps
```

---

## Error recovery

| Error | Action |
|---|---|
| `FileNotFoundError: google_credentials.json` | Stop. Tell user to download OAuth JSON from Google Cloud Console. |
| `HttpError 401` | Delete `google_token.pickle` and re-run to trigger fresh OAuth. |
| `HttpError 403` | API not enabled or scope missing — report the API name to user. |
| `HttpError 404` on an ID | Re-run the list/search tool to get a fresh ID. |
| `GOOGLE_MAPS_API_KEY not set` | Stop Maps tools. Report the missing `.env` key. |
| `Maps status: REQUEST_DENIED` | API key invalid or Maps API not enabled in Cloud Console. |

---

## DONE summary template

```
DONE: <one-line description of what was accomplished>

Meet link    : https://meet.google.com/xxx-yyyy-zzz      (if applicable)
Calendar     : https://calendar.google.com/...           (if applicable)
Drive file   : https://drive.google.com/...              (if applicable)
Time         : 10 Aug 2025, 3:00 PM - 4:00 PM IST       (human readable)
Invited      : alice@gmail.com, bob@gmail.com            (if any)
```

Always convert ISO datetimes back to human-readable IST format in the summary.
Always include every shareable link — the user needs to copy them.