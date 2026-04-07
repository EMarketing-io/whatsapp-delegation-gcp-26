from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # WhatsApp (maytapi)
    wa_product_id: str
    wa_token: str
    wa_phone_id: str
    wa_base_url: str = "https://gate.whapi.cloud"

    # OpenAI
    openai_api_key: str

    # Google Sheets
    google_sheet_id: str                          # ← paste your Sheet ID here in .env
    google_service_account_json: str = ""         # path to JSON file (local dev)
    google_service_account_json_content: str = "" # full JSON string (Render env var)

    # Google Drive (optional — voice uploads)
    google_drive_folder_id: str = ""

    # App
    frontend_url: str = "*"


settings = Settings()
