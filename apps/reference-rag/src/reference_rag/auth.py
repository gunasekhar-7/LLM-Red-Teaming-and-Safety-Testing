"""Authentication for reference RAG."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reference_rag.config import settings
from reference_rag.database import Tenant, async_session_factory


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"rag_{secrets.token_urlsafe(32)}"


def create_access_token(tenant_id: str, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_expiry_minutes)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {"sub": tenant_id, "exp": expire}
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


async def verify_token(token: str) -> str | None:
    """Verify JWT token and return tenant_id."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None


async def get_tenant_from_api_key(api_key: str) -> Tenant | None:
    """Look up tenant by API key."""
    if not settings.auth_enabled:
        # Return default tenant for testing
        return Tenant(id="default", name="Default", api_key_hash="", is_active=True)
    
    key_hash = hash_api_key(api_key)
    async with async_session_factory() as session:
        stmt = select(Tenant).where(
            Tenant.api_key_hash == key_hash,
            Tenant.is_active == True,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def get_current_tenant(
    api_key: str | None = Security(api_key_header),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> Tenant:
    """Get current tenant from API key or JWT."""
    if not settings.auth_enabled:
        return Tenant(id="default", name="Default", api_key_hash="", is_active=True)
    
    # Try API key first
    if api_key:
        tenant = await get_tenant_from_api_key(api_key)
        if tenant:
            return tenant
    
    # Try JWT
    if credentials:
        tenant_id = await verify_token(credentials.credentials)
        if tenant_id:
            async with async_session_factory() as session:
                stmt = select(Tenant).where(
                    Tenant.id == tenant_id,
                    Tenant.is_active == True,
                )
                result = await session.execute(stmt)
                tenant = result.scalar_one_or_none()
                if tenant:
                    return tenant
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_tenant(tenant: Tenant = Depends(get_current_tenant)) -> Tenant:
    """Dependency that requires a valid tenant."""
    return tenant