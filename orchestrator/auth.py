"""Login + team accounts API. The browser sends `Authorization: Bearer <token>` with every request.

    POST   /auth/login      email + password      -> token + user
    POST   /auth/logout
    GET    /auth/me                               -> the logged-in user
    POST   /auth/password   current + new         -> change your own password
    GET    /users                                 -> the whole team (any logged-in user)
    POST   /users           admin only            -> add an account (lead or member, with a first password)
    PUT    /users/{id}      admin only            -> edit skills, capacity, role, reset password ...
    DELETE /users/{id}      admin only

Who may call what: require_login (anyone logged in), require_planner (lead or admin: planning + Trello),
require_admin (accounts). PLANPILOT_AUTH=0 switches the checks off (used by the tests of the agents).
"""
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from orchestrator import db

router = APIRouter(tags=["auth"])
GUEST = {"id": 0, "email": "", "name": "auth disabled", "role": "admin"}


# ------------------------------------------------------------------ who is calling
def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    if os.getenv("PLANPILOT_AUTH", "1") == "0":
        return GUEST
    token = (authorization or "").removeprefix("Bearer ").strip()
    user = db.user_for_token(token) if token else None
    if not user:
        raise HTTPException(401, detail=[{"loc": ["header"], "msg": "Please log in again.", "type": "auth_error"}])
    return user


def require_login(user: dict = Depends(current_user)) -> dict:
    return user


def _require(*roles):
    def check(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(403, detail=[{"loc": ["header"], "type": "auth_error",
                                              "msg": f"Only {' or '.join(roles)} accounts can do this."}])
        return user
    return check


require_planner = _require("lead", "admin")
require_admin = _require("admin")


def _bad(e: Exception) -> HTTPException:
    return HTTPException(422, detail=[{"loc": ["body"], "msg": str(e), "type": "value_error"}])


# ------------------------------------------------------------------ bodies
class LoginBody(BaseModel):
    email: str
    password: str


class PasswordBody(BaseModel):
    current_password: str
    new_password: str


class NewUser(BaseModel):
    email: str
    name: str
    password: str
    role: str = "member"
    skills: List[str] = Field(default_factory=list)
    capacity_points: int = Field(default=10, ge=1)
    current_load: int = Field(default=0, ge=0)
    trello_username: Optional[str] = None
    in_team: bool = True


class EditUser(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None          # admin resets someone's password
    skills: Optional[List[str]] = None
    capacity_points: Optional[int] = Field(default=None, ge=1)
    current_load: Optional[int] = Field(default=None, ge=0)
    trello_username: Optional[str] = None
    in_team: Optional[bool] = None


# ------------------------------------------------------------------ login
@router.post("/auth/login")
def login(body: LoginBody):
    token = db.login(body.email, body.password)
    if not token:
        raise HTTPException(401, detail=[{"loc": ["body"], "msg": "Wrong email or password.", "type": "auth_error"}])
    return {"token": token, "user": db.user_for_token(token)}


@router.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    db.logout((authorization or "").removeprefix("Bearer ").strip())
    return {"ok": True}


@router.get("/auth/me")
def me(user: dict = Depends(require_login)):
    return user


@router.post("/auth/password")
def change_password(body: PasswordBody, user: dict = Depends(require_login)):
    with db.connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
    if not row or not db.check_password(body.current_password, row["password_hash"]):
        raise _bad("The current password is wrong.")
    try:
        db.update_user(user["id"], password=body.new_password)
    except ValueError as e:
        raise _bad(e)
    return {"ok": True}


# ------------------------------------------------------------------ team accounts
@router.get("/users")
def users(_: dict = Depends(require_login)):
    return db.list_users()


@router.post("/users")
def add_user(body: NewUser, _: dict = Depends(require_admin)):
    try:
        return db.create_user(**body.model_dump())
    except ValueError as e:
        raise _bad(e)


@router.put("/users/{user_id}")
def edit_user(user_id: int, body: EditUser, me_: dict = Depends(require_admin)):
    target = db.get_user(user_id)
    if not target:
        raise HTTPException(404, detail=[{"loc": ["path"], "msg": "No such account.", "type": "value_error"}])
    if target["role"] == "admin" and body.role and body.role != "admin" and db.count_admins() == 1:
        raise _bad("This is the only admin: create another admin first.")
    try:
        return db.update_user(user_id, **body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise _bad(e)


@router.delete("/users/{user_id}")
def remove_user(user_id: int, me_: dict = Depends(require_admin)):
    if user_id == me_["id"]:
        raise _bad("You cannot delete your own account.")
    db.delete_user(user_id)
    return {"ok": True}
