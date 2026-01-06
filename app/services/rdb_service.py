import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.clients import openai_client
from app.core.config import settings

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

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not settings.EMP_DB_DSN:
        raise RuntimeError("EMP_DB_DSN이 설정되지 않았습니다. 직원 DB DSN을 .env에 설정하세요.")
    return create_engine(settings.EMP_DB_DSN, pool_pre_ping=True)


def _schema_summary() -> str:
    """
    간단한 테이블/컬럼 목록을 문자열로 반환 (SQL 생성 보조용).
    """
    engine = get_engine()
    insp = inspect(engine)
    lines: List[str] = []
    for table in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(table)]
        lines.append(f"{table}({', '.join(cols)})")
    return "\n".join(lines)


def _generate_select_sql(question: str, schema: str, com_id: Optional[str]) -> str:
    """
    LLM으로 안전한 SELECT 쿼리를 생성합니다. DDL/DML 금지.
    """
    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = s[3:].lstrip()  # remove leading ```
            if s.lower().startswith("sql"):
                s = s[3:].lstrip()  # remove optional 'sql'
            if s.endswith("```"):
                s = s[:-3].rstrip()
        return s

    extra = f"com_id 컬럼이 존재하면 WHERE com_id = '{com_id}' 조건을 반드시 포함하세요. " if com_id else ""
    prompt = (
        "다음 질문을 SQL SELECT 한 개로 변환하세요. 테이블/컬럼은 스키마에 명시된 것만 사용합니다. "
        "INSERT/UPDATE/DELETE/DDL은 금지. LIMIT 20 이하로 설정하세요. 이름/텍스트 검색은 LIKE '%키워드%'를 사용하세요. "
        f"{extra}"
        "답변은 코드펜스 없이 SQL만 출력하고, 세미콜론은 붙이지 마세요.\n\n"
        f"[스키마]\n{schema}\n\n[질문]\n{question}"
    )
    resp = openai_client.chat.completions.create(
        model=settings.RDB_MODEL,
        messages=[{"role": "system", "content": "You are a SQL assistant that only writes safe read-only queries."},
                  {"role": "user", "content": prompt}],
        temperature=0,
    )
    raw_sql = (resp.choices[0].message.content or "").strip()
    sql = _strip_code_fence(raw_sql)
    # 마지막 세미콜론은 제거해도 무방
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    print(f"[RDB] generated SQL: {sql}")
    return sql


def _is_safe_select(sql: str) -> bool:
    lowered = sql.lower()
    # 내부에서 세미콜론을 허용하지 않음(문장 분리로 오인)
    if ";" in lowered:
        return False
    if not lowered.lstrip().startswith("select"):
        return False
    banned = ["insert", "update", "delete", "drop", "alter", "truncate"]
    return not any(f" {b} " in lowered for b in banned)


def _ensure_limit(sql: str, default_limit: int = 50) -> str:
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {default_limit}"


def _extract_tables(sql: str) -> List[str]:
    # 매우 단순한 FROM/JOIN 테이블명 추출
    tbls = re.findall(r"(?:from|join)\s+([`\"\w]+)", sql, flags=re.IGNORECASE)
    return [t.strip("`\"").lower() for t in tbls]


def _ensure_allowed_tables(sql: str):
    tables = _extract_tables(sql)
    if not tables:
        return
    for t in tables:
        if t not in ALLOWED_TABLES:
            raise RuntimeError(f"허용되지 않은 테이블 접근 시도: {t}")


def execute_select(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    SELECT만 실행. 결과를 dict 리스트로 반환.
    """
    if not _is_safe_select(sql):
        raise RuntimeError("허용되지 않는 SQL입니다. SELECT만 지원합니다.")

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
        return [dict(r) for r in rows]


def _ensure_com_filter(sql: str, com_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    com_id가 주어졌는데 SQL에 com_id 조건이 없다면 강제로 추가합니다.
    """
    params: Dict[str, Any] = {}
    if not com_id:
        return sql, params
    lowered = sql.lower()
    if "com_id" not in lowered:
        if " where " in lowered:
            sql = f"{sql} AND com_id = :com_id"
        else:
            sql = f"{sql} WHERE com_id = :com_id"
    params["com_id"] = com_id
    return sql, params


def _ensure_personal_filter(sql: str, tables: List[str], emp_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if not emp_id:
        return sql, params
    if not any(t in PERSONAL_TABLES for t in tables):
        return sql, params
    lowered = sql.lower()
    if "emp_id" not in lowered:
        if " where " in lowered:
            sql = f"{sql} AND emp_id = :emp_id"
        else:
            sql = f"{sql} WHERE emp_id = :emp_id"
    params["emp_id"] = emp_id
    return sql, params


def query_db_with_llm(question: str, com_id: Optional[str], emp_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    질문을 SQL로 변환 후 실행. 결과 반환.
    """
    schema = _schema_summary()
    sql = _generate_select_sql(question, schema, com_id)
    sql = _ensure_limit(sql)
    _ensure_allowed_tables(sql)
    tables = _extract_tables(sql)
    sql, params = _ensure_com_filter(sql, com_id)
    # 개인 테이블 필터 병합
    p_sql, p_params = _ensure_personal_filter(sql, tables, emp_id)
    params.update(p_params)
    return execute_select(p_sql, params)


def query_employee_contact_by_name(name_keyword: str, com_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    간단한 이름 키워드로 직원 연락처를 조회합니다.
    """
    if not com_id:
        raise RuntimeError("com_id가 필요합니다.")
    engine = get_engine()
    sql = text(
        """
        SELECT emp_name, email, phone, work_phone, dep_no, pos_no, emp_id, com_id
        FROM employee
        WHERE com_id = :com_id
          AND emp_name LIKE :name_like
        LIMIT 20
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"com_id": com_id, "name_like": f"%{name_keyword}%"}).mappings().all()
        return [dict(r) for r in rows]
