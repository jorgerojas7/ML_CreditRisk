from fastapi import APIRouter, Depends
from app.auth.dependencies import (
    require_bank_agent,
    require_client,
    require_any_role,
)
from app.users.models import User

router = APIRouter(prefix="/predictions", tags=["Examples - Protected Routes"])

@router.get("/public")
def public_endpoint():
    return {
        "message": "Este es un endpoint público",
        "access": "No requiere autenticación",
        "info": "Cualquier persona puede acceder a esta ruta"
    }

@router.get("/bank-agent-only")
def bank_agent_only_endpoint(current_user: User = Depends(require_bank_agent)):
    return {
        "message": f"Bienvenido agente {current_user.full_name}",
        "access": "Solo Bank Agents",
        "user_info": {
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role.value
        },
        "available_actions": [
            "Realizar predicciones de crédito",
            "Validar predicciones",
            "Ver todas las predicciones",
            "Generar reportes"
        ]
    }

@router.get("/client-only")
def client_only_endpoint(current_user: User = Depends(require_client)):
    return {
        "message": f"Bienvenido {current_user.full_name}",
        "access": "Solo Clients",
        "user_info": {
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role.value
        },
        "available_actions": [
            "Ver mis predicciones",
            "Ver historial de solicitudes",
            "Actualizar perfil"
        ]
    }

@router.get("/authenticated-users")
def authenticated_users_endpoint(current_user: User = Depends(require_any_role)):
    return {
        "message": f"Hola {current_user.full_name}",
        "access": "Cualquier usuario autenticado",
        "user_info": {
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role.value,
            "is_active": current_user.is_active
        },
        "info": "Este endpoint es accesible para cualquier usuario con token válido"
    }