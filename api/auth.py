import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from src.config import BASE_DIR


JWT_ALGORITHM = "HS256"
JWT_ISSUER = "bmce-credit-risk-api"
JWT_AUDIENCE = "bmce-credit-risk-users"
PASSWORD_ITERATIONS = 600_000
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,40}$")
bearer_scheme = HTTPBearer(auto_error=False)


class Role(str, Enum):
    admin = "admin"
    analyste = "analyste"
    conseiller = "conseiller"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=128)
    role: Role = Role.conseiller

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        value = value.strip()
        if not USERNAME_PATTERN.fullmatch(value):
            raise ValueError("Le nom doit contenir 3 à 40 lettres, chiffres, points, tirets ou underscores.")
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if value.count("@") != 1 or "." not in value.split("@", 1)[1]:
            raise ValueError("Adresse e-mail invalide.")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not re.search(r"[a-z]", value):
            raise ValueError("Le mot de passe doit contenir une minuscule.")
        if not re.search(r"[A-Z]", value):
            raise ValueError("Le mot de passe doit contenir une majuscule.")
        if not re.search(r"\d", value):
            raise ValueError("Le mot de passe doit contenir un chiffre.")
        return value


class BootstrapRequest(UserCreate):
    role: Role = Role.admin


class RoleUpdate(BaseModel):
    role: Role


class ActiveUpdate(BaseModel):
    is_active: bool


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: Role
    is_active: bool
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


auth_router = APIRouter(prefix="/auth", tags=["Authentification"])
admin_router = APIRouter(prefix="/admin", tags=["Administration"])


def get_database_path() -> Path:
    configured = os.getenv("AUTH_DB_PATH")
    return Path(configured) if configured else BASE_DIR / "data" / "auth.db"


def get_connection() -> sqlite3.Connection:
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_auth_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'analyste', 'conseiller')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if secret:
        return secret
    if environment == "production":
        raise RuntimeError("JWT_SECRET_KEY doit être définie en production.")
    return "development-only-secret-change-before-production"


def access_token_minutes() -> int:
    return int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "30"))


def create_access_token(user: sqlite3.Row | dict) -> tuple[str, int]:
    minutes = access_token_minutes()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)
    return token, minutes * 60


def user_to_response(user: sqlite3.Row | dict) -> UserResponse:
    return UserResponse(
        id=int(user["id"]),
        username=str(user["username"]),
        email=str(user["email"]),
        role=Role(user["role"]),
        is_active=bool(user["is_active"]),
        created_at=str(user["created_at"]),
    )


def find_user_by_username(username: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()


def find_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def insert_user(payload: UserCreate, forced_role: Role | None = None) -> sqlite3.Row:
    created_at = datetime.now(timezone.utc).isoformat()
    role = forced_role or payload.role
    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (payload.username, payload.email, hash_password(payload.password), role.value, created_at),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Ce nom d'utilisateur ou cet e-mail existe déjà.") from exc
    user = find_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=500, detail="Impossible de relire l'utilisateur créé.")
    return user


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    access_token: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> sqlite3.Row:
    token = credentials.credentials if credentials else access_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification requise.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Jeton invalide ou expiré.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    user = find_user_by_id(user_id)
    if user is None or not bool(user["is_active"]):
        raise HTTPException(status_code=401, detail="Compte inexistant ou désactivé.")
    return user


CurrentUser = Annotated[sqlite3.Row, Depends(get_current_user)]


def require_roles(*allowed_roles: Role) -> Callable:
    async def role_checker(current_user: CurrentUser) -> sqlite3.Row:
        if Role(current_user["role"]) not in allowed_roles:
            raise HTTPException(status_code=403, detail="Permissions insuffisantes.")
        return current_user

    return role_checker


AdminUser = Annotated[sqlite3.Row, Depends(require_roles(Role.admin))]


@auth_router.get("/status")
def auth_status() -> dict[str, bool]:
    init_auth_db()
    with get_connection() as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    return {"initialized": count > 0}


@auth_router.post("/bootstrap", response_model=UserResponse, status_code=201)
def bootstrap_admin(payload: BootstrapRequest) -> UserResponse:
    init_auth_db()
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        count = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        if count > 0:
            raise HTTPException(status_code=409, detail="L'administrateur initial existe déjà.")
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO users (username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, 'admin', 1, ?)
            """,
            (payload.username, payload.email, hash_password(payload.password), created_at),
        )
        user_id = int(cursor.lastrowid)
    user = find_user_by_id(user_id)
    return user_to_response(user)


@auth_router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, response: Response) -> TokenResponse:
    user = find_user_by_username(payload.username)
    if user is None or not bool(user["is_active"]) or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    token, expires_in = create_access_token(user)
    response.set_cookie(
        key="access_token",
        value=token,
        max_age=expires_in,
        httponly=True,
        secure=os.getenv("ENVIRONMENT", "development").lower() == "production",
        samesite="lax",
        path="/",
    )
    return TokenResponse(access_token=token, expires_in=expires_in, user=user_to_response(user))


@auth_router.post("/logout", status_code=204)
def logout(response: Response) -> Response:
    response.delete_cookie("access_token", path="/")
    return response


@auth_router.get("/me", response_model=UserResponse)
def me(current_user: CurrentUser) -> UserResponse:
    return user_to_response(current_user)


@admin_router.get("/users", response_model=list[UserResponse])
def list_users(_: AdminUser) -> list[UserResponse]:
    with get_connection() as connection:
        users = connection.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [user_to_response(user) for user in users]


@admin_router.post("/users", response_model=UserResponse, status_code=201)
def create_user(payload: UserCreate, _: AdminUser) -> UserResponse:
    return user_to_response(insert_user(payload))


@admin_router.patch("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(user_id: int, payload: RoleUpdate, admin: AdminUser) -> UserResponse:
    if user_id == int(admin["id"]) and payload.role != Role.admin:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas retirer votre propre rôle administrateur.")
    with get_connection() as connection:
        cursor = connection.execute("UPDATE users SET role = ? WHERE id = ?", (payload.role.value, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return user_to_response(find_user_by_id(user_id))


@admin_router.patch("/users/{user_id}/active", response_model=UserResponse)
def update_user_active(user_id: int, payload: ActiveUpdate, admin: AdminUser) -> UserResponse:
    if user_id == int(admin["id"]) and not payload.is_active:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas désactiver votre propre compte.")
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (int(payload.is_active), user_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return user_to_response(find_user_by_id(user_id))
