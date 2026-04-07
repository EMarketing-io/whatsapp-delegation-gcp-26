"""
Google Sheets service.

Tabs:
  Tasks        — A:R  (main task data)
  Message Logs — A:F  (raw log)
  Config       — A: Name, B: Mail ID, D: Customer Name, E: Departments
"""
import json
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import settings

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

TASK_COLUMNS = [
    "timestamp", "task_id", "task_description", "assigned_by",
    "assignee_contact", "assigned_to", "employee_email_id", "target_date",
    "priority", "approval_needed", "client_name", "department",
    "assigned_name", "assigned_email_id", "comments", "source_link",
    "status", "message_type", "updated_timestamp",
]

TASK_DISPLAY_NAMES = {
    "task_description":  "Task Description",
    "assigned_to":       "Assigned To",
    "employee_email_id": "Employee Email",
    "target_date":       "Target Date",
    "priority":          "Priority",
    "approval_needed":   "Approval Needed",
    "client_name":       "Client Name",
    "department":        "Department",
    "assigned_name":     "Assigned Name",
    "assigned_email_id": "Assigned Email",
    "comments":          "Comments",
}


def _get_service():
    if settings.google_service_account_json_content:
        info = json.loads(settings.google_service_account_json_content)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file(
            settings.google_service_account_json, scopes=SCOPES
        )
    return build("sheets", "v4", credentials=creds)


# ── Config sheet lookup ───────────────────────────────────────────────────────

def get_config_lookup() -> dict:
    """
    Returns:
      {
        "employees":     { name_lower: email },
        "employee_names":{ name_lower: display_name },
        "customers":     ["Acme Corp", ...],
        "departments":   ["Google Ads", "Meta Ads", ...]
      }
    """
    service = _get_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=settings.google_sheet_id, range="Config!A2:E")
        .execute()
    )
    rows = result.get("values", [])
    employees = {}
    employee_names = {}
    customers = []
    departments = []
    for row in rows:
        row += [""] * (5 - len(row))
        name, email, _, customer, department = row[0], row[1], row[2], row[3], row[4]
        if name and email:
            key = name.strip().lower()
            employees[key] = email.strip()
            employee_names[key] = name.strip()
        if customer:
            customers.append(customer.strip())
        if department:
            departments.append(department.strip())
    return {"employees": employees, "employee_names": employee_names, "customers": customers, "departments": departments}


def lookup_customer_name(mentioned: str, config: dict) -> tuple[str, bool]:
    """
    Match a partial/full client name against Config Customer Names.
    Returns (full_name, matched):
      - ("Bikaner Polymers Pvt Ltd", True)  → partial match found in Config
      - ("Singhdaur", False)                → mentioned but no match in Config
      - ("", False)                         → nothing mentioned
    """
    if not mentioned or not mentioned.strip():
        return "", False
    needle = mentioned.strip().lower()
    for customer in config["customers"]:
        if needle in customer.lower() or customer.lower() in needle:
            return customer, True
    return "", False  # no match — leave blank, caller will warn user


def lookup_department_name(mentioned: str, config: dict) -> tuple[str, bool]:
    """
    Match a partial/full department name against Config Departments (col E).
    Returns (full_name, matched):
      - ("Google Ads", True)  → partial match found
      - ("", False)           → no match — leave blank, warn user
      - ("", False)           → nothing mentioned
    """
    if not mentioned or not mentioned.strip():
        return "", False
    needle = mentioned.strip().lower()
    for dept in config["departments"]:
        if needle in dept.lower() or dept.lower() in needle:
            return dept, True
    return "", False  # no match — leave blank, caller will warn user


def _find_employee(name: str, config: dict) -> tuple[str, str]:
    """
    Returns (full_name, email) for the best matching employee.
    Tries exact match first, then partial match.
    Returns ("", "") if no match found.
    """
    if not name:
        return "", ""
    needle = name.strip().lower()
    employees = config["employees"]  # {full_name_lower: email}
    full_names = config["employee_names"]  # {full_name_lower: display_name}

    # 1. Exact match
    if needle in employees:
        return full_names[needle], employees[needle]

    # 2. Partial match
    for full_name_lower, email in employees.items():
        if needle in full_name_lower or full_name_lower in needle:
            return full_names[full_name_lower], email

    return "", ""


def lookup_employee_full_name(name: str, config: dict) -> tuple[str, bool]:
    """
    Returns (full_name, matched).
    - ("Taaran Jain", True)  → found in Config
    - ("", False)            → not found — leave blank, caller warns user
    - ("", False)            → nothing provided
    """
    full_name, _ = _find_employee(name, config)
    if full_name:
        return full_name, True
    return "", False


def lookup_employee_email(name: str, config: dict) -> str:
    _, email = _find_employee(name, config)
    return email


# ── Task ID ───────────────────────────────────────────────────────────────────

def get_next_task_id() -> str:
    service = _get_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=settings.google_sheet_id, range="Tasks!B:B")
        .execute()
    )
    count = len(result.get("values", []))
    return f"TASK-{count:04d}"


# ── Tasks ─────────────────────────────────────────────────────────────────────

def append_task(task_data: dict) -> None:
    service = _get_service()
    row = [str(task_data.get(col, "") or "") for col in TASK_COLUMNS]
    (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=settings.google_sheet_id,
            range="Tasks!A:R",
            valueInputOption="USER_ENTERED",
            body={"values": [row]},
        )
        .execute()
    )


def get_all_tasks(status: str = None, priority: str = None, limit: int = 100, offset: int = 0) -> list[dict]:
    service = _get_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=settings.google_sheet_id, range="Tasks!A2:R")
        .execute()
    )
    rows = result.get("values", [])
    tasks = []
    for row in rows:
        row += [""] * (len(TASK_COLUMNS) - len(row))
        task = dict(zip(TASK_COLUMNS, row))
        if status and task.get("status") != status:
            continue
        if priority and task.get("priority") != priority:
            continue
        tasks.append(task)
    tasks.reverse()
    return tasks[offset: offset + limit]


def get_task_by_id(task_id: str) -> dict | None:
    tasks = get_all_tasks()
    for t in tasks:
        if t.get("task_id") == task_id:
            return t
    return None


def update_task(task_id: str, updates: dict) -> dict | None:
    service = _get_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=settings.google_sheet_id, range="Tasks!B:B")
        .execute()
    )
    rows = result.get("values", [])
    row_index = None
    for i, row in enumerate(rows):
        if row and row[0] == task_id:
            row_index = i + 1
            break
    if row_index is None:
        return None

    # Always stamp updated_timestamp on any update
    updates["updated_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for col_name, value in updates.items():
        if col_name in TASK_COLUMNS:
            col_letter = chr(ord("A") + TASK_COLUMNS.index(col_name))
            (
                service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=settings.google_sheet_id,
                    range=f"Tasks!{col_letter}{row_index}",
                    valueInputOption="USER_ENTERED",
                    body={"values": [[str(value)]]},
                )
                .execute()
            )

    return get_task_by_id(task_id)


def mark_task_done(task_id: str) -> dict | None:
    """Set status to Done and stamp updated_timestamp."""
    return update_task(task_id, {"status": "Done"})


# ── Confirmation message builder ──────────────────────────────────────────────

def build_confirmation_message(task: dict) -> str:
    """Build the WhatsApp reply showing filled and pending fields."""
    filled = []
    pending = []

    for col, label in TASK_DISPLAY_NAMES.items():
        val = task.get(col, "")
        if val and val.strip():
            filled.append(f"  • {label}: {val}")
        else:
            pending.append(f"  • {label}")

    lines = [f"✅ Task Recorded! ID: *{task['task_id']}*", ""]

    if filled:
        lines.append("📋 *Details Recorded:*")
        lines.extend(filled)

    if pending:
        lines.append("")
        lines.append("⏳ *Pending Details:*")
        lines.extend(pending)
        lines.append("")
        lines.append(f"To fill pending details, reply:\n*/update {task['task_id']}*\nfollowed by the missing info in natural language.")
        lines.append(f"\nExample:\n/update {task['task_id']} department: Marketing, email: john@acme.com, approval: yes")

    return "\n".join(lines)


# ── Message Logs ──────────────────────────────────────────────────────────────

def add_client_to_config(client_name: str) -> bool:
    """
    Append a new customer name to col D of the Config sheet.
    Finds the next empty row in col D and writes there.
    Returns False if client already exists.
    """
    service = _get_service()

    # Check if already exists
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=settings.google_sheet_id, range="Config!D:D")
        .execute()
    )
    existing = [r[0].strip().lower() for r in result.get("values", []) if r]
    if client_name.strip().lower() in existing:
        return False  # already exists

    # Find next empty row in col D
    next_row = len(result.get("values", [])) + 1

    (
        service.spreadsheets()
        .values()
        .update(
            spreadsheetId=settings.google_sheet_id,
            range=f"Config!D{next_row}",
            valueInputOption="USER_ENTERED",
            body={"values": [[client_name.strip()]]},
        )
        .execute()
    )
    return True


def log_message(sender: str, msg_type: str, raw_text: str, task_id: str = "", error: str = "") -> None:
    service = _get_service()
    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        sender,
        msg_type,
        raw_text[:500],
        task_id,
        error,
    ]
    (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=settings.google_sheet_id,
            range="Message Logs!A:F",
            valueInputOption="USER_ENTERED",
            body={"values": [row]},
        )
        .execute()
    )
