"""
tools/google_tools.py — Google Workspace tools for Shifu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Covers: Google Drive, Google Calendar, Google Meet (via Calendar),
        Google Maps (Places + Directions via Maps API).

Auth: OAuth 2.0 for Drive/Calendar/Meet  |  API Key for Maps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import io
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from langchain_core.tools import tool

# ─── Config ───────────────────────────────────────────────────────────────────

# All Google OAuth scopes we need
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/calendar",
]

# Paths (relative to project root — same folder as shifu.py)
_CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", "google_credentials.json"))
_TOKEN_FILE       = Path(os.getenv("GOOGLE_TOKEN_FILE",       "google_token.pickle"))
_MAPS_API_KEY     = os.getenv("GOOGLE_MAPS_API_KEY", "")


# ─── Auth helpers ──────────────────────────────────────────────────────────────

def _get_google_creds() -> Credentials:
    """Load or refresh OAuth2 credentials, prompting the user if needed."""
    creds: Optional[Credentials] = None

    if _TOKEN_FILE.exists():
        with open(_TOKEN_FILE, "rb") as fh:
            creds = pickle.load(fh)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Google credentials file not found at '{_CREDENTIALS_FILE}'. "
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(_TOKEN_FILE, "wb") as fh:
            pickle.dump(creds, fh)

    return creds


def _drive_service():
    return build("drive", "v3", credentials=_get_google_creds())


def _calendar_service():
    return build("calendar", "v3", credentials=_get_google_creds())


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def gdrive_list_files(query: str = "", max_results: int = 20) -> str:
    """
    List files in Google Drive.

    Args:
        query: Optional Drive search query (e.g. "name contains 'report'" or
               "mimeType='application/pdf'"). Leave empty to list recent files.
        max_results: Maximum number of files to return (default 20).

    Returns:
        JSON string with a list of {id, name, mimeType, modifiedTime, webViewLink}.
    """
    try:
        svc = _drive_service()
        params: dict = {
            "pageSize": max_results,
            "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
            "orderBy": "modifiedTime desc",
        }
        if query:
            params["q"] = query
        result = svc.files().list(**params).execute()
        files = result.get("files", [])
        return json.dumps(files, indent=2)
    except HttpError as e:
        return f"Error listing Drive files: {e}"


@tool
def gdrive_upload_file(local_path: str, drive_folder_id: str = "", mime_type: str = "") -> str:
    """
    Upload a local file to Google Drive.

    Args:
        local_path: Absolute or relative path to the local file.
        drive_folder_id: Optional Drive folder ID to upload into. Leave empty for root.
        mime_type: MIME type of the file (e.g. 'application/pdf'). Auto-detected if empty.

    Returns:
        JSON with {id, name, webViewLink} of the uploaded file.
    """
    try:
        path = Path(local_path)
        if not path.exists():
            return f"Error: file not found at '{local_path}'"

        import mimetypes
        detected_mime, _ = mimetypes.guess_type(str(path))
        final_mime = mime_type or detected_mime or "application/octet-stream"

        metadata: dict = {"name": path.name}
        if drive_folder_id:
            metadata["parents"] = [drive_folder_id]

        media = MediaFileUpload(str(path), mimetype=final_mime, resumable=True)
        svc = _drive_service()
        f = svc.files().create(
            body=metadata, media_body=media,
            fields="id,name,webViewLink"
        ).execute()
        return json.dumps(f, indent=2)
    except HttpError as e:
        return f"Error uploading to Drive: {e}"


@tool
def gdrive_download_file(file_id: str, save_to: str = "") -> str:
    """
    Download a file from Google Drive to the local Playground/ folder.

    Args:
        file_id: The Drive file ID.
        save_to: Local path to save to (default: Playground/<original_name>).

    Returns:
        Success message with local path, or error string.
    """
    try:
        svc = _drive_service()
        meta = svc.files().get(fileId=file_id, fields="name,mimeType").execute()
        file_name = meta["name"]
        dest = Path(save_to) if save_to else Path("Playground") / file_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        request = svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        dest.write_bytes(buf.getvalue())
        return f"✅ Downloaded '{file_name}' → {dest.resolve()}"
    except HttpError as e:
        return f"Error downloading from Drive: {e}"


@tool
def gdrive_create_folder(folder_name: str, parent_folder_id: str = "") -> str:
    """
    Create a folder in Google Drive.

    Args:
        folder_name: Name for the new folder.
        parent_folder_id: Optional parent folder ID. Leave empty for root.

    Returns:
        JSON with {id, name, webViewLink}.
    """
    try:
        metadata: dict = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_folder_id:
            metadata["parents"] = [parent_folder_id]
        svc = _drive_service()
        folder = svc.files().create(body=metadata, fields="id,name,webViewLink").execute()
        return json.dumps(folder, indent=2)
    except HttpError as e:
        return f"Error creating Drive folder: {e}"


@tool
def gdrive_share_file(file_id: str, email: str, role: str = "reader") -> str:
    """
    Share a Google Drive file with a specific user.

    Args:
        file_id: The Drive file ID.
        email: Email address of the person to share with.
        role: Permission role — 'reader', 'commenter', or 'writer' (default: 'reader').

    Returns:
        Success message or error string.
    """
    try:
        svc = _drive_service()
        permission = {"type": "user", "role": role, "emailAddress": email}
        svc.permissions().create(
            fileId=file_id, body=permission, sendNotificationEmail=True
        ).execute()
        return f"✅ Shared file {file_id} with {email} as {role}."
    except HttpError as e:
        return f"Error sharing Drive file: {e}"


@tool
def gdrive_delete_file(file_id: str) -> str:
    """
    Permanently delete a file from Google Drive.

    Args:
        file_id: The Drive file ID.

    Returns:
        Success message or error string.
    """
    try:
        _drive_service().files().delete(fileId=file_id).execute()
        return f"✅ Deleted Drive file {file_id}."
    except HttpError as e:
        return f"Error deleting Drive file: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE CALENDAR TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def gcalendar_list_events(
    calendar_id: str = "primary",
    days_ahead: int = 7,
    max_results: int = 20,
) -> str:
    """
    List upcoming events from Google Calendar.

    Args:
        calendar_id: Calendar ID (default 'primary' for main calendar).
        days_ahead: How many days ahead to look (default 7).
        max_results: Maximum events to return (default 20).

    Returns:
        JSON list of events with id, summary, start, end, hangoutLink.
    """
    try:
        svc = _calendar_service()
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        result = svc.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = result.get("items", [])
        simplified = [
            {
                "id":          e.get("id"),
                "summary":     e.get("summary", "(no title)"),
                "start":       e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                "end":         e.get("end",   {}).get("dateTime") or e.get("end",   {}).get("date"),
                "location":    e.get("location", ""),
                "description": e.get("description", ""),
                "hangoutLink": e.get("hangoutLink", ""),
                "htmlLink":    e.get("htmlLink", ""),
            }
            for e in events
        ]
        return json.dumps(simplified, indent=2)
    except HttpError as e:
        return f"Error listing Calendar events: {e}"


@tool
def gcalendar_create_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    attendee_emails: str = "",
    calendar_id: str = "primary",
) -> str:
    """
    Create a Google Calendar event.

    Args:
        title: Event title / summary.
        start_datetime: ISO 8601 start time, e.g. '2025-08-10T10:00:00+05:30'.
        end_datetime:   ISO 8601 end time,   e.g. '2025-08-10T11:00:00+05:30'.
        description: Optional event description.
        location: Optional location string.
        attendee_emails: Comma-separated email addresses to invite.
        calendar_id: Calendar to add to (default 'primary').

    Returns:
        JSON with event id, htmlLink, and a confirmation message.
    """
    try:
        svc = _calendar_service()
        attendees = [
            {"email": e.strip()}
            for e in attendee_emails.split(",")
            if e.strip()
        ]
        body: dict = {
            "summary":     title,
            "description": description,
            "location":    location,
            "start":       {"dateTime": start_datetime},
            "end":         {"dateTime": end_datetime},
            "attendees":   attendees,
        }
        event = svc.events().insert(calendarId=calendar_id, body=body,
                                    sendUpdates="all").execute()
        return json.dumps({
            "id":       event["id"],
            "htmlLink": event.get("htmlLink"),
            "message":  f"✅ Event '{title}' created.",
        }, indent=2)
    except HttpError as e:
        return f"Error creating Calendar event: {e}"


@tool
def gcalendar_update_event(
    event_id: str,
    title: str = "",
    start_datetime: str = "",
    end_datetime: str = "",
    description: str = "",
    location: str = "",
    calendar_id: str = "primary",
) -> str:
    """
    Update an existing Google Calendar event (only provided fields are changed).

    Args:
        event_id: The event ID to update.
        title: New title (optional).
        start_datetime: New ISO 8601 start time (optional).
        end_datetime:   New ISO 8601 end time (optional).
        description: New description (optional).
        location: New location (optional).
        calendar_id: Calendar the event belongs to (default 'primary').

    Returns:
        Success message or error string.
    """
    try:
        svc = _calendar_service()
        event = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
        if title:          event["summary"]     = title
        if description:    event["description"] = description
        if location:       event["location"]    = location
        if start_datetime: event["start"]       = {"dateTime": start_datetime}
        if end_datetime:   event["end"]         = {"dateTime": end_datetime}
        updated = svc.events().update(
            calendarId=calendar_id, eventId=event_id, body=event,
            sendUpdates="all"
        ).execute()
        return f"✅ Event '{updated.get('summary')}' updated. Link: {updated.get('htmlLink')}"
    except HttpError as e:
        return f"Error updating Calendar event: {e}"


@tool
def gcalendar_delete_event(event_id: str, calendar_id: str = "primary") -> str:
    """
    Delete a Google Calendar event.

    Args:
        event_id: The event ID to delete.
        calendar_id: Calendar the event belongs to (default 'primary').

    Returns:
        Success message or error string.
    """
    try:
        _calendar_service().events().delete(
            calendarId=calendar_id, eventId=event_id, sendUpdates="all"
        ).execute()
        return f"✅ Event {event_id} deleted."
    except HttpError as e:
        return f"Error deleting Calendar event: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE MEET TOOLS  (Meet links are created via Calendar)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def gmeet_create_meeting(
    title: str,
    start_datetime: str,
    end_datetime: str,
    attendee_emails: str = "",
    description: str = "",
    calendar_id: str = "primary",
) -> str:
    """
    Create a Google Meet meeting and return the join link.
    The meeting is backed by a Google Calendar event with a Meet conference.

    Args:
        title: Meeting title.
        start_datetime: ISO 8601 start time, e.g. '2025-08-10T15:00:00+05:30'.
        end_datetime:   ISO 8601 end time,   e.g. '2025-08-10T16:00:00+05:30'.
        attendee_emails: Comma-separated emails to invite (they receive the Meet link).
        description: Optional meeting agenda / description.
        calendar_id: Calendar to use (default 'primary').

    Returns:
        JSON with {event_id, meet_link, calendar_link, title, attendees}.
    """
    try:
        svc = _calendar_service()
        attendees = [
            {"email": e.strip()}
            for e in attendee_emails.split(",")
            if e.strip()
        ]
        body: dict = {
            "summary":     title,
            "description": description,
            "start":       {"dateTime": start_datetime},
            "end":         {"dateTime": end_datetime},
            "attendees":   attendees,
            "conferenceData": {
                "createRequest": {
                    "requestId": f"shifu-meet-{int(datetime.now().timestamp())}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }
        event = svc.events().insert(
            calendarId=calendar_id,
            body=body,
            conferenceDataVersion=1,
            sendUpdates="all",
        ).execute()

        meet_link = event.get("hangoutLink", "")
        return json.dumps({
            "event_id":      event["id"],
            "meet_link":     meet_link,
            "calendar_link": event.get("htmlLink", ""),
            "title":         title,
            "attendees":     [a["email"] for a in attendees],
            "message": (
                f"✅ Meet '{title}' created. "
                f"Join: {meet_link}"
            ),
        }, indent=2)
    except HttpError as e:
        return f"Error creating Meet meeting: {e}"


@tool
def gmeet_get_link_for_event(event_id: str, calendar_id: str = "primary") -> str:
    """
    Retrieve the Google Meet join link for an existing Calendar event.

    Args:
        event_id: The Calendar event ID.
        calendar_id: Calendar the event belongs to (default 'primary').

    Returns:
        The Meet link string, or an error message.
    """
    try:
        svc = _calendar_service()
        event = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
        link = event.get("hangoutLink", "")
        if link:
            return f"Meet link for '{event.get('summary', event_id)}': {link}"
        return "No Meet link found for this event (it may not have a conference attached)."
    except HttpError as e:
        return f"Error fetching Meet link: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE MAPS TOOLS
#  Places Search & Details  →  Places API (New) — v1 REST
#  Directions & Geocoding   →  unchanged (these APIs were NOT deprecated)
# ═══════════════════════════════════════════════════════════════════════════════

def _maps_key() -> str:
    key = _MAPS_API_KEY or os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY is not set. Add it to your .env file."
        )
    return key


# New Places API base URL
_PLACES_V1 = "https://places.googleapis.com/v1"


def _places_headers() -> dict:
    return {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   _maps_key(),
    }


@tool
def gmaps_search_places(query: str, location: str = "", radius_meters: int = 5000) -> str:
    """
    Search for places using the Google Places API (New) Text Search.

    Args:
        query: Search query, e.g. 'coffee shops' or 'AIIMS Hospital Patna'.
        location: Optional 'lat,lng' to bias results, e.g. '25.5941,85.1376'.
        radius_meters: Bias radius in metres when location is provided (default 5000).

    Returns:
        JSON list of places with name, address, rating, place_id, maps_url.
    """
    try:
        body: dict = {"textQuery": query}
        if location:
            lat, lng = location.split(",")
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": float(lat), "longitude": float(lng)},
                    "radius": float(radius_meters),
                }
            }

        headers = _places_headers()
        # Tell the API which fields to return (billing-aware field mask)
        headers["X-Goog-FieldMask"] = (
            "places.id,places.displayName,places.formattedAddress,"
            "places.rating,places.regularOpeningHours.openNow,"
            "places.googleMapsUri"
        )

        resp = requests.post(
            f"{_PLACES_V1}/places:searchText",
            headers=headers,
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        places = data.get("places", [])
        results = [
            {
                "name":      p.get("displayName", {}).get("text"),
                "address":   p.get("formattedAddress"),
                "rating":    p.get("rating"),
                "place_id":  p.get("id"),
                "maps_url":  p.get("googleMapsUri"),
                "open_now":  p.get("regularOpeningHours", {}).get("openNow"),
            }
            for p in places[:10]
        ]
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Maps: {e}"


@tool
def gmaps_get_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
) -> str:
    """
    Get directions between two places using Google Maps Directions API.

    Args:
        origin: Starting point — address or 'lat,lng'.
        destination: Destination — address or 'lat,lng'.
        mode: Travel mode — 'driving', 'walking', 'bicycling', or 'transit' (default 'driving').

    Returns:
        JSON with distance, duration, and step-by-step directions.
    """
    try:
        params = {
            "origin":      origin,
            "destination": destination,
            "mode":        mode,
            "key":         _maps_key(),
        }
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            return f"Directions API error: {data.get('status')} — {data.get('error_message', '')}"

        leg   = data["routes"][0]["legs"][0]
        steps = [
            {
                "instruction": s["html_instructions"],
                "distance":    s["distance"]["text"],
                "duration":    s["duration"]["text"],
            }
            for s in leg["steps"]
        ]
        return json.dumps({
            "origin":      leg["start_address"],
            "destination": leg["end_address"],
            "distance":    leg["distance"]["text"],
            "duration":    leg["duration"]["text"],
            "mode":        mode,
            "steps":       steps,
        }, indent=2)
    except Exception as e:
        return f"Error getting directions: {e}"


@tool
def gmaps_geocode(address: str) -> str:
    """
    Convert a human-readable address to latitude/longitude coordinates.

    Args:
        address: The address to geocode, e.g. 'Patna Junction, Bihar, India'.

    Returns:
        JSON with formatted_address, lat, lng, and place_id.
    """
    try:
        params = {"address": address, "key": _maps_key()}
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            return f"Geocoding error: {data.get('status')} — {data.get('error_message', '')}"

        r   = data["results"][0]
        loc = r["geometry"]["location"]
        return json.dumps({
            "formatted_address": r["formatted_address"],
            "lat":      loc["lat"],
            "lng":      loc["lng"],
            "place_id": r["place_id"],
        }, indent=2)
    except Exception as e:
        return f"Error geocoding address: {e}"


@tool
def gmaps_place_details(place_id: str) -> str:
    """
    Get detailed information about a place using the Google Places API (New).

    Args:
        place_id: The place ID returned by gmaps_search_places.

    Returns:
        JSON with name, address, phone, website, hours, rating, reviews.
    """
    try:
        headers = _places_headers()
        headers["X-Goog-FieldMask"] = (
            "id,displayName,formattedAddress,internationalPhoneNumber,"
            "websiteUri,regularOpeningHours.weekdayDescriptions,rating,reviews"
        )

        resp = requests.get(
            f"{_PLACES_V1}/places/{place_id}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        r = resp.json()

        hours = r.get("regularOpeningHours", {}).get("weekdayDescriptions", [])
        reviews = [
            {
                "author": rv.get("authorAttribution", {}).get("displayName"),
                "rating": rv.get("rating"),
                "text":   rv.get("text", {}).get("text", "")[:200],
            }
            for rv in r.get("reviews", [])[:3]
        ]
        return json.dumps({
            "name":    r.get("displayName", {}).get("text"),
            "address": r.get("formattedAddress"),
            "phone":   r.get("internationalPhoneNumber"),
            "website": r.get("websiteUri"),
            "hours":   hours,
            "rating":  r.get("rating"),
            "reviews": reviews,
        }, indent=2)
    except Exception as e:
        return f"Error fetching place details: {e}"