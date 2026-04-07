import json
from datetime import date
from pathlib import Path

from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)

EXTRACTION_PROMPT = """
You are a task extraction assistant for a delegation app.
Today's date is {today}.
Given the raw message below, extract the following fields and return ONLY valid JSON.
If a field cannot be determined, use null.

Fields to extract:
- task_description: string — concise summary of what needs to be done
- assigned_to: string — person's name the task is for
- employee_email_id: string — assignee's email if explicitly mentioned
- target_date: string — deadline in YYYY-MM-DD format. Resolve relative dates like "today", "tomorrow", "next Monday", "after 5 days", "end of week" using today's date. Use null only if no date is mentioned at all.
- priority: one of ["Low", "Medium", "High", "Critical"]
- approval_needed: boolean — true if approval/sign-off is mentioned
- client_name: string — client or company name if mentioned (partial names are fine, write exactly as mentioned)
- department: string — department if mentioned
- assigned_name: string — delegator's name if different from sender
- assigned_email_id: string — delegator's email if explicitly mentioned
- comments: string — ALWAYS generate a brief contextual note about this task based on what was said, even if not explicitly stated. Example: "Rahul to update the bot flow and deploy by tomorrow."

Raw message:
\"\"\"
{message}
\"\"\"

Respond with ONLY the JSON object, no markdown, no explanation.
"""


async def transcribe_audio(audio_path: str) -> tuple[str, str]:
    with open(audio_path, "rb") as audio_file:
        original = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    with open(audio_path, "rb") as audio_file:
        translated = await client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    return str(original).strip(), str(translated).strip()


async def extract_task_fields(raw_message: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(today=date.today().isoformat(), message=raw_message)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


UPDATE_PROMPT = """
You are a task update assistant. The user is filling in missing details for a task.
Today's date is {today}.
Extract only the fields they are providing and return ONLY valid JSON.
Use null for any field not mentioned.

Updatable fields:
- employee_email_id: string — assignee's email
- department: string
- approval_needed: "Yes" or "No"
- comments: string
- target_date: string — YYYY-MM-DD format. Resolve relative dates like "today", "tomorrow", "next Monday", "after 5 days" using today's date.
- priority: one of ["Low", "Medium", "High", "Critical"]
- assigned_to: string — assignee name
- client_name: string
- assigned_name: string
- assigned_email_id: string

User message:
\"\"\"{message}\"\"\"

Respond with ONLY the JSON object, no markdown, no explanation.
"""


async def extract_update_fields(raw_message: str) -> dict:
    """Extract field updates from a /update message using GPT-4o."""
    prompt = UPDATE_PROMPT.format(today=date.today().isoformat(), message=raw_message)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    # Remove null values — only keep what was actually provided
    return {k: v for k, v in data.items() if v is not None}
