from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List
from functools import wraps

from app.database import get_db
from app.auth.utils import decode_access_token
from app.users.repository import UserRepository
from app.users.models import User, UserRole

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Dependency para obtener el usuario actual desde el token JWT"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = UserRepository.get_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    
    if not UserRepository.is_active(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario inactivo"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency para obtener usuario activo"""
    return current_user

class RoleChecker:
    """Clase para verificar roles de usuario - DECORADOR"""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operación no permitida. Se requiere uno de estos roles: {[role.value for role in self.allowed_roles]}"
            )
        return current_user

require_bank_agent = RoleChecker([UserRole.BANK_AGENT])

require_client = RoleChecker([UserRole.CLIENT])

require_any_role = RoleChecker([UserRole.BANK_AGENT, UserRole.CLIENT])