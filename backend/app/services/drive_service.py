import json
import mimetypes
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from app.config import settings

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_drive_service():
    if settings.google_service_account_json_content:
        info = json.loads(settings.google_service_account_json_content)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            settings.google_service_account_json, scopes=SCOPES
        )
    return build("drive", "v3", credentials=creds)


def upload_audio_to_drive(local_path: str, filename: str) -> str:
    service = _get_drive_service()

    mime_type = mimetypes.guess_type(filename)[0] or "audio/ogg"

    file_metadata = {
        "name": filename,
        "parents": [settings.google_drive_folder_id],
    }
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=False)

    uploaded = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    file_id = uploaded["id"]

    service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"},
        supportsAllDrives=True,
    ).execute()

    return f"https://drive.google.com/file/d/{file_id}/view"
