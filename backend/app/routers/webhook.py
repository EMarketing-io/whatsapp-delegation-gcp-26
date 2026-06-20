import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.services import openai_service, drive_service, sheets_service, db_service

logger = logging.getLogger("webhook")
router = APIRouter()

HELP_MESSAGE = """🤖 *WhatsApp Delegation Bot — Commands*

📌 *Create a Task*
`/task <description>`
_Assign a task to someone. Mention name, deadline, priority, client._
Example: `/task give website redesign to Rahul by Friday, high priority, client: Acme Corp`

✅ *Mark Task Done*
`/done TASK-0001`
_Changes task status to Done and records the timestamp._

🔄 *Update a Task*
`/update TASK-0001 <details>`
_Fill in any pending/blank fields on an existing task._
Example: `/update TASK-0001 department: Marketing, approval: yes`

📋 *Check Task Status*
`/status TASK-0001`
_Get the current details of a specific task._

📂 *My Pending Tasks*
`/my-tasks`
_See all tasks currently assigned to you that are not yet done._

➕ *Add a Client*
`/add-client <name>`
_Add a new customer name to the Config sheet._
Example: `/add-client Acme Corp`"""


def _verify_secret(request: Request) -> None:
    """Log headers for debugging; secret verification disabled until header name is confirmed."""
    safe_headers = {
        k: ("***" if settings.waumfy_webhook_secret and settings.waumfy_webhook_secret in v else v)
        for k, v in request.headers.items()
    }
    logger.info("Incoming headers: %s", json.dumps(safe_headers))


def _parse_waumfy_event(payload: dict) -> tuple[dict, str, str, str]:
    """
    Parse a Waumfy webhook payload.

    Returns (data, sender_phone, sender_name, chat_id).
    Raises ValueError if the payload isn't a MESSAGE_RECEIVED event.

    Waumfy now includes chatType: 'individual' or 'group' in every payload.
    Use data.from directly as chat_id — Waumfy's send API handles
    @s.whatsapp.net, @lid, and @g.us routing internally (normalizeJid).
    """
    if payload.get("event") != "MESSAGE_RECEIVED":
        raise ValueError(f"ignored event: {payload.get('event')}")

    data = payload.get("data", {})
    if not data:
        raise ValueError("no data in payload")

    if data.get("fromMe"):
        raise ValueError("ignoring own outgoing message (fromMe=true)")

    from_jid = str(data.get("from", "")).strip()
    raw_phone = str(data.get("senderPhone", "")).strip().lstrip("+")
    chat_type = data.get("chatType", "")  # "individual" or "group"

    # senderPhone for group messages is sometimes a WhatsApp internal ID (15-digit, starts with 120).
    # The real phone number is embedded in the participant JID: "919876543210@s.whatsapp.net"
    participant_jid = str(data.get("participant", "")).strip()
    if participant_jid and "@" in participant_jid:
        participant_phone = participant_jid.split("@")[0].lstrip("+")
    else:
        participant_phone = ""

    if raw_phone.startswith("120") and len(raw_phone) >= 12 and participant_phone:
        raw_phone = participant_phone

    # Use from JID directly — Waumfy normalizes @s.whatsapp.net, @lid, @g.us on their end
    chat_id = from_jid

    sender_phone = f"+{raw_phone}" if raw_phone else ""
    sender_name = data.get("senderName") or sender_phone
    logger.info("chat_type=%s from=%s senderPhone=%s senderName=%s", chat_type, from_jid, raw_phone, sender_name)
    return data, sender_phone, sender_name, chat_id


async def _send_reply(chat_id: str, text: str) -> None:
    """Send a text message via Waumfy outgoing trigger API, with retries for DNS/network failures."""
    if not settings.waumfy_send_url:
        logger.warning("WAUMFY_SEND_URL not set — reply not sent")
        return
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    settings.waumfy_send_url,
                    headers={
                        "X-API-Key": settings.waumfy_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"to": chat_id, "text": text},
                )
                logger.info("Reply to %s → status=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                resp.raise_for_status()
                return
        except Exception as exc:
            logger.warning("Reply attempt %d/3 failed for %s: %s", attempt + 1, chat_id, exc)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s
    logger.error("All reply attempts failed for %s", chat_id)


async def _process_text(
    raw_text: str, sender: str, sender_name: str
) -> tuple[dict, str]:
    config = sheets_service.get_config_lookup()
    fields = await openai_service.extract_task_fields(raw_text)

    raw_assigned_to = fields.get("assigned_to", "")
    assigned_to, assignee_matched = sheets_service.lookup_employee_full_name(
        raw_assigned_to, config
    )
    employee_email = fields.get(
        "employee_email_id"
    ) or sheets_service.lookup_employee_email(assigned_to, config)
    assignee_warning = (
        ""
        if assignee_matched or not raw_assigned_to
        else (
            f"\n⚠️ Assignee *{raw_assigned_to}* was not found in the Config list. "
            f"Please reply:\n`/update {{TASK_ID}} assigned_to: <correct name>`\nto fix this task."
        )
    )

    assigned_name, _ = sheets_service.lookup_employee_full_name(
        fields.get("assigned_name") or sender_name, config
    )
    assigned_name = assigned_name or sender_name
    assigned_email = fields.get(
        "assigned_email_id"
    ) or sheets_service.lookup_employee_email(sender_name, config)
    raw_client = fields.get("client_name", "")
    client_name, client_matched = sheets_service.lookup_customer_name(
        raw_client, config
    )
    client_warning = (
        ""
        if client_matched or not raw_client
        else (
            f"\n⚠️ Client *{raw_client}* was not found in the Config list. "
            f"Please use `/add-client <correct name>` to add it, then reply:\n"
            f"`/update {{TASK_ID}} client: <correct client name>`\nto fix this task."
        )
    )

    raw_dept = fields.get("department", "")
    department, dept_matched = sheets_service.lookup_department_name(raw_dept, config)
    dept_warning = (
        ""
        if dept_matched or not raw_dept
        else (
            f"\n⚠️ Department *{raw_dept}* was not found in the Config list. "
            f"Please reply:\n`/update {{TASK_ID}} department: <correct department>`\nto fix this task."
        )
    )

    task_id = sheets_service.get_next_task_id()
    task_data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "task_id": task_id,
        "task_description": fields.get("task_description", ""),
        "assigned_by": sender_name,
        "assignee_contact": sender,
        "assigned_to": assigned_to,
        "employee_email_id": employee_email,
        "target_date": fields.get("target_date", ""),
        "priority": fields.get("priority", "Medium"),
        "approval_needed": "Yes" if fields.get("approval_needed") else "No",
        "client_name": client_name,
        "department": department,
        "assigned_name": assigned_name,
        "assigned_email_id": assigned_email,
        "comments": fields.get("comments", ""),
        "source_link": "",
        "status": "Pending",
        "message_type": raw_text,
    }
    sheets_service.append_task(task_data)
    await db_service.insert_task(task_data)
    warning = (assignee_warning + client_warning + dept_warning).replace(
        "{TASK_ID}", task_id
    )
    return task_data, warning


async def _process_voice(media_url: str, sender: str, sender_name: str) -> tuple[dict, str]:
    ext = ".ogg"
    for candidate in [".ogg", ".mp3", ".mp4", ".m4a", ".wav", ".opus"]:
        if candidate in media_url.lower():
            ext = candidate
            break

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(
            media_url,
            headers={"X-API-Key": settings.waumfy_api_key},
        )
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()

    try:
        original_text, english_text = await openai_service.transcribe_audio(tmp.name)
        logger.info(
            "Voice transcription: %r | translation: %r", original_text, english_text
        )

        drive_url = media_url
        if settings.google_drive_folder_id:
            try:
                filename = (
                    f"voice_{sender}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{ext}"
                )
                drive_url = drive_service.upload_audio_to_drive(tmp.name, filename)
                logger.info("Uploaded to Drive: %s", drive_url)
            except Exception as drive_err:
                logger.warning(
                    "Drive upload failed, saving WhatsApp URL instead: %s", drive_err
                )

        config = sheets_service.get_config_lookup()
        fields = await openai_service.extract_task_fields(english_text)

        raw_assigned_to = fields.get("assigned_to", "")
        assigned_to, assignee_matched = sheets_service.lookup_employee_full_name(
            raw_assigned_to, config
        )
        employee_email = fields.get(
            "employee_email_id"
        ) or sheets_service.lookup_employee_email(assigned_to, config)
        assignee_warning = (
            ""
            if assignee_matched or not raw_assigned_to
            else (
                f"\n⚠️ Assignee *{raw_assigned_to}* was not found in the Config list. "
                f"Please reply:\n`/update {{TASK_ID}} assigned_to: <correct name>`\nto fix this task."
            )
        )

        assigned_name, _ = sheets_service.lookup_employee_full_name(
            fields.get("assigned_name") or sender_name, config
        )
        assigned_name = assigned_name or sender_name
        assigned_email = fields.get(
            "assigned_email_id"
        ) or sheets_service.lookup_employee_email(sender_name, config)
        raw_client = fields.get("client_name", "")
        client_name, client_matched = sheets_service.lookup_customer_name(
            raw_client, config
        )
        client_warning = (
            ""
            if client_matched or not raw_client
            else (
                f"\n⚠️ Client *{raw_client}* was not found in the Config list. "
                f"Please use `/add-client <correct name>` to add it, then reply:\n"
                f"`/update {{TASK_ID}} client: <correct client name>`\nto fix this task."
            )
        )

        raw_dept = fields.get("department", "")
        department, dept_matched = sheets_service.lookup_department_name(
            raw_dept, config
        )
        dept_warning = (
            ""
            if dept_matched or not raw_dept
            else (
                f"\n⚠️ Department *{raw_dept}* was not found in the Config list. "
                f"Please reply:\n`/update {{TASK_ID}} department: <correct department>`\nto fix this task."
            )
        )

        task_id = sheets_service.get_next_task_id()
        task_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "task_description": fields.get("task_description", ""),
            "assigned_by": sender_name,
            "assignee_contact": sender,
            "assigned_to": assigned_to,
            "employee_email_id": employee_email,
            "target_date": fields.get("target_date", ""),
            "priority": fields.get("priority", "Medium"),
            "approval_needed": "Yes" if fields.get("approval_needed") else "No",
            "client_name": client_name,
            "department": department,
            "assigned_name": assigned_name,
            "assigned_email_id": assigned_email,
            "comments": fields.get("comments", ""),
            "source_link": drive_url,
            "status": "Pending",
            "message_type": (
                f"[Voice] {original_text}"
                if original_text == english_text
                else f"[Voice] {original_text} | [EN] {english_text}"
            ),
        }
        sheets_service.append_task(task_data)
        await db_service.insert_task(task_data)
        warning = (assignee_warning + client_warning + dept_warning).replace(
            "{TASK_ID}", task_id
        )
        return task_data, warning
    finally:
        os.unlink(tmp.name)


async def _process_done(raw_text: str, sender: str, chat_id: str) -> None:
    match = re.search(r"(TASK-\d+)", raw_text, re.IGNORECASE)
    if not match:
        await _send_reply(chat_id, "❌ Could not find a Task ID. Use format:\n/done TASK-0001")
        return

    task_id = match.group(1).upper()
    result = sheets_service.mark_task_done(task_id)
    await db_service.mark_task_done(task_id)

    if result is None:
        await _send_reply(chat_id, f"❌ Task {task_id} not found.")
        return

    await _send_reply(chat_id, f"✅ *{task_id}* marked as *Done!*")


async def _process_update(raw_text: str, sender: str, chat_id: str) -> None:
    match = re.search(r"(TASK-\d+)", raw_text, re.IGNORECASE)
    if not match:
        await _send_reply(
            chat_id,
            "❌ Could not find a Task ID. Use format:\n/update TASK-0001 department: Marketing, email: john@acme.com",
        )
        return

    task_id = match.group(1).upper()
    update_text = raw_text[match.end():].strip()

    if not update_text:
        await _send_reply(
            chat_id,
            f"❌ No update info provided. Example:\n/update {task_id} department: Marketing, email: john@acme.com",
        )
        return

    updates = await openai_service.extract_update_fields(update_text)

    if not updates:
        await _send_reply(
            chat_id,
            f"❌ Could not understand the update. Example:\n/update {task_id} department: Marketing, email: john@acme.com",
        )
        return

    result = sheets_service.update_task(task_id, updates)
    await db_service.update_task(task_id, updates)

    if result is None:
        await _send_reply(chat_id, f"❌ Task {task_id} not found.")
        return

    confirmation = sheets_service.build_confirmation_message(result)
    updated_fields = ", ".join(updates.keys())
    await _send_reply(
        chat_id,
        f"✅ *{task_id}* updated!\nFields updated: {updated_fields}\n\n{confirmation}",
    )


@router.get("/webhook")
async def webhook_verify(request: Request):
    logger.info("Webhook verification GET received")
    return {"status": "ok"}


@router.post("/webhook")
async def webhook(request: Request):
    _verify_secret(request)

    payload = await request.json()
    logger.info("WEBHOOK RECEIVED: %s", json.dumps(payload, indent=2))

    try:
        data, sender, sender_name, chat_id = _parse_waumfy_event(payload)
    except ValueError as e:
        logger.info("Ignored event: %s", e)
        return {"status": "ignored"}

    msg_type = data.get("type", "")
    logger.info("msg_type=%s sender=%s chat_id=%s", msg_type, sender, chat_id)

    task_data = None
    warning = ""
    error = None

    try:
        if msg_type == "text":
            body: str = data.get("body", "")
            logger.info("Text body: %r", body)

            # ── Explicit slash commands ──────────────────────────
            if body.lower().strip() == "/help":
                await _send_reply(chat_id, HELP_MESSAGE)
                return {"status": "ok"}

            elif body.lower().startswith("/task"):
                task_data, warning = await _process_text(body, sender, sender_name)

            elif body.lower().startswith("/status"):
                match = re.search(r"(TASK-\d+)", body, re.IGNORECASE)
                if not match:
                    await _send_reply(chat_id, "❌ Usage: /status TASK-0001")
                else:
                    task = sheets_service.get_task_by_id(match.group(1).upper())
                    if not task:
                        await _send_reply(chat_id, f"❌ Task {match.group(1).upper()} not found.")
                    else:
                        await _send_reply(chat_id, sheets_service.build_confirmation_message(task))
                return {"status": "ok"}

            elif body.lower().strip() == "/my-tasks":
                all_tasks = sheets_service.get_all_tasks()
                my_tasks = [
                    t
                    for t in all_tasks
                    if (
                        t.get("assignee_contact") == sender
                        or t.get("assignee_contact") == f"+{sender}"
                    )
                    and t.get("status", "").lower()
                    not in ("done", "completed", "cancelled")
                ]
                if not my_tasks:
                    await _send_reply(chat_id, "✅ You have no pending tasks!")
                else:
                    lines = [f"📋 *Your Pending Tasks ({len(my_tasks)})*\n"]
                    for t in my_tasks:
                        lines.append(
                            f"• *{t.get('task_id')}* — {t.get('task_description') or 'No description'}\n"
                            f"  Priority: {t.get('priority') or '—'} | Due: {t.get('target_date') or '—'} | Status: {t.get('status') or '—'}"
                        )
                    await _send_reply(chat_id, "\n".join(lines))
                return {"status": "ok"}

            elif body.lower().startswith("/add-client"):
                client_name = body[len("/add-client"):].strip()
                if not client_name:
                    await _send_reply(
                        chat_id,
                        "❌ Please provide a client name.\nExample: `/add-client Acme Corp`",
                    )
                else:
                    added = sheets_service.add_client_to_config(client_name)
                    if added:
                        await _send_reply(chat_id, f"✅ Client *{client_name}* added to Config successfully!")
                    else:
                        await _send_reply(chat_id, f"⚠️ Client *{client_name}* already exists in Config.")
                sheets_service.log_message(sender, msg_type, body, "", "")
                await db_service.log_message(sender, msg_type, body, sender_name=sender_name)
                return {"status": "ok"}

            elif body.lower().startswith("/done"):
                await _process_done(body, sender, chat_id)
                sheets_service.log_message(sender, msg_type, body, "", "")
                await db_service.log_message(sender, msg_type, body, sender_name=sender_name)
                return {"status": "ok"}

            elif body.lower().startswith("/update"):
                await _process_update(body, sender, chat_id)
                sheets_service.log_message(sender, msg_type, body, "", "")
                await db_service.log_message(sender, msg_type, body, sender_name=sender_name)
                return {"status": "ok"}

            # ── Plain natural language — classify with AI ────────
            else:
                intent_data = await openai_service.classify_intent(body)
                intent = intent_data.get("intent", "other")
                task_id = (intent_data.get("task_id") or "").upper()
                logger.info("AI intent=%s task_id=%s", intent, task_id)

                if intent == "task":
                    task_data, warning = await _process_text(body, sender, sender_name)

                elif intent == "done":
                    if task_id:
                        await _process_done(f"/done {task_id}", sender, chat_id)
                    else:
                        await _send_reply(chat_id, "Which task is done? Reply with:\n`/done TASK-0001`")
                    sheets_service.log_message(sender, msg_type, body, task_id, "")
                    await db_service.log_message(sender, msg_type, body, task_id, sender_name=sender_name)
                    return {"status": "ok"}

                elif intent == "update":
                    if task_id:
                        await _process_update(f"/update {task_id} {body}", sender, chat_id)
                    else:
                        await _send_reply(chat_id, "Which task to update? Reply with:\n`/update TASK-0001 <details>`")
                    sheets_service.log_message(sender, msg_type, body, task_id, "")
                    await db_service.log_message(sender, msg_type, body, task_id, sender_name=sender_name)
                    return {"status": "ok"}

                elif intent == "status":
                    if task_id:
                        task = sheets_service.get_task_by_id(task_id)
                        if task:
                            await _send_reply(chat_id, sheets_service.build_confirmation_message(task))
                        else:
                            await _send_reply(chat_id, f"❌ Task {task_id} not found.")
                    else:
                        await _send_reply(chat_id, "Which task? Reply with:\n`/status TASK-0001`")
                    sheets_service.log_message(sender, msg_type, body, task_id, "")
                    await db_service.log_message(sender, msg_type, body, task_id, sender_name=sender_name)
                    return {"status": "ok"}

                elif intent == "my_tasks":
                    all_tasks = sheets_service.get_all_tasks()
                    my_tasks = [
                        t for t in all_tasks
                        if (
                            t.get("assignee_contact") == sender
                            or t.get("assignee_contact") == f"+{sender}"
                        )
                        and t.get("status", "").lower() not in ("done", "completed", "cancelled")
                    ]
                    if not my_tasks:
                        await _send_reply(chat_id, "✅ You have no pending tasks!")
                    else:
                        lines = [f"📋 *Your Pending Tasks ({len(my_tasks)})*\n"]
                        for t in my_tasks:
                            lines.append(
                                f"• *{t.get('task_id')}* — {t.get('task_description') or 'No description'}\n"
                                f"  Priority: {t.get('priority') or '—'} | Due: {t.get('target_date') or '—'} | Status: {t.get('status') or '—'}"
                            )
                        await _send_reply(chat_id, "\n".join(lines))
                    sheets_service.log_message(sender, msg_type, body, "", "")
                    await db_service.log_message(sender, msg_type, body, sender_name=sender_name)
                    return {"status": "ok"}

                elif intent == "help":
                    await _send_reply(chat_id, HELP_MESSAGE)
                    return {"status": "ok"}

                else:
                    # "other" — casual chat, ignore silently
                    logger.info("Ignoring non-task message (intent=other)")
                    return {"status": "ignored"}

        elif msg_type in ("audio", "ptt", "voice"):
            media_url = data.get("url") or data.get("mediaUrl") or data.get("link")
            if media_url:
                task_data, warning = await _process_voice(media_url, sender, sender_name)

    except Exception as exc:
        logger.exception("Error processing message: %s", exc)
        error = str(exc)
        await _send_reply(chat_id, f"❌ Error processing your message: {exc}")

    raw = data.get("body") or data.get("url") or ""
    tid = task_data.get("task_id", "") if task_data else ""
    sheets_service.log_message(sender=sender, msg_type=msg_type, raw_text=raw, task_id=tid, error=error or "")
    await db_service.log_message(sender, msg_type, raw, tid, error or "", sender_name=sender_name)

    if task_data:
        confirmation = sheets_service.build_confirmation_message(task_data)
        if warning:
            confirmation += warning
        await _send_reply(chat_id, confirmation)

    return {"status": "ok"}
