"""
CRUD endpoints consumed by the Next.js frontend.
Reads/writes Google Sheets instead of a database.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services import sheets_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[str] = None
    employee_email_id: Optional[str] = None
    target_date: Optional[str] = None
    comments: Optional[str] = None
    approval_needed: Optional[str] = None
    department: Optional[str] = None
    client_name: Optional[str] = None


@router.get("/")
def list_tasks(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    return sheets_service.get_all_tasks(status=status, priority=priority, limit=limit, offset=offset)


@router.get("/{task_id}")
def get_task(task_id: str):
    tasks = sheets_service.get_all_tasks()
    for task in tasks:
        if task.get("task_id") == task_id:
            return task
    raise HTTPException(status_code=404, detail="Task not found")


@router.patch("/{task_id}")
def update_task(task_id: str, body: TaskUpdate):
    updates = body.model_dump(exclude_none=True)
    result = sheets_service.update_task(task_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result
