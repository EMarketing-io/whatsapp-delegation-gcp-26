from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str

    # Google Sheets
    google_sheet_id: str
    google_service_account_json: str = ""  # path to JSON file (local dev)
    google_service_account_json_content: str = ""  # full JSON string (Cloud Run / Render)

    # Google Drive (optional — voice uploads)
    google_drive_folder_id: str = ""

    # Waumfy — incoming webhook verification
    waumfy_webhook_secret: str = ""

    # Waumfy — outgoing API (send replies)
    waumfy_send_url: str = ""   # https://api.aumpfy.com/api/apis/trigger/<slug>
    waumfy_api_key: str = ""    # X-API-Key header value

    # MySQL Database
    db_host: str = ""
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""

    # App
    frontend_url: str = "*"


settings = Settings()
