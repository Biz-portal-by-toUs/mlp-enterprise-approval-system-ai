from datetime import datetime
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from app.services.rdb_service import (
    get_engine,
    query_db_with_llm,
    query_employee_contact_by_name,
)


ALLOWED_TABLES = {
    "approval_line",
    "attendance",
    "board",
    "corporate_car",
    "corporate_car_reservation",
    "meeting_room",
    "meeting_room_reservation",
    "shared_equipment",
    "shared_equipment_reservation",
    "emp_schedule",
    "employee",
    "todo_list",
    "mail",
    "meeting",
    "meeting_emp",
    "schedule",
}

PERSONAL_TABLES = {"todo_list", "mail", "attendance", "emp_schedule"}


def _limit_clause(limit: int | None, default: int = 50) -> str:
    lim = limit or default
    return f" LIMIT {min(lim, default)}"


def get_my_todos(emp_id: str, com_id: str, status: str | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT todo_no, title, is_done, created_at, updated_at "
        "FROM todo_list WHERE com_id = :com_id AND emp_id = :emp_id"
    )
    params = {"com_id": com_id, "emp_id": emp_id}
    if status:
        sql += " AND is_done = :status"
        params["status"] = status
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_my_mails(emp_id: str, com_id: str, unread_only: bool = False, limit: int | None = None) -> List[Dict[str, Any]]:
    # 제한: 발신한 메일만 조회 (mail_user_state는 화이트리스트에 없어 사용 불가)
    sql = (
        "SELECT mail_no, title, created_at, sender_id "
        "FROM mail WHERE sender_id = :emp_id"
    )
    params = {"emp_id": emp_id}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_my_attendance(emp_id: str, com_id: str, start: datetime | None = None, end: datetime | None = None) -> List[Dict[str, Any]]:
    sql = "SELECT day, type, created_at FROM attendance WHERE com_id = :com_id AND emp_id = :emp_id"
    params = {"com_id": com_id, "emp_id": emp_id}
    if start:
        sql += " AND day >= :start"
        params["start"] = start
    if end:
        sql += " AND day <= :end"
        params["end"] = end
    sql += _limit_clause(None)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_meetings_for_me(emp_id: str, com_id: str, start: datetime | None = None, end: datetime | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT m.meet_no, m.title, m.started_at, m.status "
        "FROM meeting m JOIN meeting_emp me ON m.meet_no = me.meet_no "
        "WHERE m.com_id = :com_id AND me.emp_id = :emp_id"
    )
    params = {"com_id": com_id, "emp_id": emp_id}
    if start:
        sql += " AND m.started_at >= :start"
        params["start"] = start
    if end:
        sql += " AND m.started_at <= :end"
        params["end"] = end
    sql += _limit_clause(None)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_meeting_room_reservations(com_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT mr.room_no, r.room_name, mr.started_at, mr.ended_at, mr.resv_emp "
        "FROM meeting_room_reservation mr "
        "LEFT JOIN meeting_room r ON mr.room_no = r.room_no "
        "WHERE mr.com_id = :com_id "
        "ORDER BY mr.started_at DESC"
    )
    params = {"com_id": com_id}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_corporate_car_reservations(com_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT cr.car_no, c.car_name, cr.started_at, cr.ended_at, cr.resv_emp "
        "FROM corporate_car_reservation cr "
        "LEFT JOIN corporate_car c ON cr.car_no = c.car_no "
        "WHERE cr.com_id = :com_id "
        "ORDER BY cr.started_at DESC"
    )
    params = {"com_id": com_id}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_shared_equipment_reservations(com_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT sr.eq_no, e.eq_name, sr.started_at, sr.ended_at, sr.resv_emp "
        "FROM shared_equipment_reservation sr "
        "LEFT JOIN shared_equipment e ON sr.eq_no = e.eq_no "
        "WHERE sr.com_id = :com_id "
        "ORDER BY sr.started_at DESC"
    )
    params = {"com_id": com_id}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def search_employee_by_name(com_id: str, name: str, limit: int | None = None) -> List[Dict[str, Any]]:
    return query_employee_contact_by_name(name, com_id)


def get_employee_contact(com_id: str, emp_id: str) -> List[Dict[str, Any]]:
    # reuse name search by emp_id exact
    engine = get_engine()
    sql = text(
        "SELECT emp_name, email, phone, work_phone, emp_id, dep_no, pos_no "
        "FROM employee WHERE com_id = :com_id AND emp_id = :emp_id LIMIT 5"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"com_id": com_id, "emp_id": emp_id}).mappings().all()
        return [dict(r) for r in rows]


def search_board_posts(com_id: str, keyword: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT board_no, title, created_at, emp_id "
        "FROM board WHERE com_id = :com_id AND (title LIKE :kw OR contents LIKE :kw)"
    )
    params = {"com_id": com_id, "kw": f"%{keyword}%"}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


def get_approval_lines(com_id: str, emp_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT apprl_no, doc_no, appr_stat, ended_at "
        "FROM approval_line WHERE com_id = :com_id AND emp_id = :emp_id "
        "ORDER BY ended_at DESC"
    )
    params = {"com_id": com_id, "emp_id": emp_id}
    sql += _limit_clause(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]


TOOL_REGISTRY = {
    "get_my_todos": get_my_todos,
    "get_my_mails": get_my_mails,
    "get_my_attendance": get_my_attendance,
    "get_meetings_for_me": get_meetings_for_me,
    "get_meeting_rooms_availability": get_meeting_room_reservations,
    "get_corporate_car_availability": get_corporate_car_reservations,
    "get_shared_equipment_availability": get_shared_equipment_reservations,
    "search_employee_by_name": search_employee_by_name,
    "get_employee_contact": get_employee_contact,
    "search_board_posts": search_board_posts,
    "get_approval_lines": get_approval_lines,
}


def execute_rdb_tasks(tasks: List[Dict[str, Any]], com_id: str | None, emp_id: str | None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    주어진 rdb task 리스트를 실행하고 결과 리스트와 간단 요약 텍스트를 반환.
    """
    all_rows: List[Dict[str, Any]] = []
    summaries: List[str] = []
    for t in tasks:
        name = t.get("name")
        args = t.get("args") or {}
        fn = TOOL_REGISTRY.get(name)
        if not fn:
            print(f"[RDB TOOL] unknown task {name}, skip")
            continue
        try:
            # com_id/emp_id 강제 주입
            if "com_id" in fn.__code__.co_varnames and com_id:
                args.setdefault("com_id", com_id)
            if "emp_id" in fn.__code__.co_varnames and emp_id:
                args.setdefault("emp_id", emp_id)
            rows = fn(**args)
            all_rows.extend(rows)
            summaries.append(f"{name}: {len(rows)} rows")
        except Exception as e:
            print(f"[RDB TOOL] {name} failed: {e}")
    return all_rows, summaries


def format_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    if not rows:
        return ""
    sample = rows[:max_rows]
    headers = list(sample[0].keys())
    lines = [" | ".join(headers)]
    for r in sample:
        line = " | ".join(str(r.get(h, "")) for h in headers)
        lines.append(line)
    if len(rows) > max_rows:
        lines.append(f"...({len(rows) - max_rows} more)")
    return "\n".join(lines)
