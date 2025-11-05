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
        "message": "This is a public endpoint accessible to everyone.",
        "access": "Public"
    }

@router.get("/bank-agent-only")
def bank_agent_only_endpoint(current_user: User = Depends(require_bank_agent)):
    return {
        "message": f"Welcome {current_user.full_name}",
        "access": "Bank Agents Only",
    }

@router.get("/client-only")
def client_only_endpoint(current_user: User = Depends(require_client)):
    return {
        "message": f"Hello {current_user.full_name}",
        "access": "Clients Only",
    }

@router.get("/authenticated-users")
def authenticated_users_endpoint(current_user: User = Depends(require_any_role)):
    return {
        "message": f"Hi {current_user.full_name}",
        "access": "Authenticated Users",
    }