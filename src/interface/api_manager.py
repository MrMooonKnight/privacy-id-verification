"""
API Manager module for handling external API requests and interfaces.
"""

import logging
import jwt
import datetime
import os
from fastapi import APIRouter, Depends, HTTPException, Security, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Pydantic models for request/response data
class UserData(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    date_of_birth: str
    address: Optional[Dict[str, str]] = None

class BiometricData(BaseModel):
    type: str
    template_data: Optional[Dict[str, Any]] = None

class DocumentData(BaseModel):
    type: str
    document_number: str
    issued_by: str
    issue_date: str
    expiration_date: str
    additional_data: Optional[Dict[str, Any]] = None

class IdentityRegistrationRequest(BaseModel):
    user_data: UserData
    biometric_data: Optional[BiometricData] = None
    document_data: Optional[DocumentData] = None

class VerificationRequest(BaseModel):
    user_id: str
    verification_type: str
    verification_data: Dict[str, Any]

class PermissionsUpdateRequest(BaseModel):
    user_id: str
    authorized_viewers: Optional[List[str]] = None
    expiration: Optional[str] = None

class TokenData(BaseModel):
    username: str
    exp: datetime.datetime

class APIManager:
    """Manages API endpoints and routes for the identity verification system."""
    
    def __init__(self, app, identity_manager, security_config):
        """
        Initialize the APIManager with the required dependencies.
        
        Args:
            app: The FastAPI application
            identity_manager: The identity manager for handling identity operations
            security_config (dict): Security configuration parameters
        """
        self.app = app
        self.identity_manager = identity_manager
        self.security_config = security_config
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        logger.info("APIManager initialized")
    
    def include_routers(self):
        """Set up and include all API routers."""
        # Create API routers
        auth_router = self._create_auth_router()
        identity_router = self._create_identity_router()
        verification_router = self._create_verification_router()
        
        # Include routers in the app
        self.app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
        self.app.include_router(identity_router, prefix="/identity", tags=["Identity Management"])
        self.app.include_router(verification_router, prefix="/verification", tags=["Identity Verification"])
        
        logger.info("API routers configured")
    
    def _create_auth_router(self):
        """Create the authentication router."""
        router = APIRouter()
        
        @router.post("/token", response_model=Dict[str, str])
        async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
            """Endpoint for user login and token generation."""
            # In a real system, this would authenticate against a user database
            # For demo purposes, we'll accept any username/password
            if not form_data.username or not form_data.password:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create JWT token
            token_expires = datetime.datetime.now() + datetime.timedelta(minutes=self.security_config["token_expiry_minutes"])
            token_data = {
                "sub": form_data.username,
                "exp": token_expires
            }
            token = jwt.encode(token_data, self.security_config["jwt_secret"], algorithm="HS256")
            
            return {"access_token": token, "token_type": "bearer"}
        
        return router
    
    def _create_identity_router(self):
        """Create the identity management router."""
        router = APIRouter()
        
        @router.post("/register", response_model=Dict[str, Any])
        async def register_identity(
            registration_data: IdentityRegistrationRequest,
            token: str = Depends(self.oauth2_scheme)
        ):
            """Register a new identity in the system."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Register identity
            result = self.identity_manager.register_identity(
                registration_data.user_data.dict(),
                registration_data.biometric_data.dict() if registration_data.biometric_data else None,
                registration_data.document_data.dict() if registration_data.document_data else None
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
        
        @router.post("/upload-biometric", response_model=Dict[str, Any])
        async def upload_biometric_data(
            user_id: str = Form(...),
            biometric_type: str = Form(...),
            biometric_file: UploadFile = File(...),
            token: str = Depends(self.oauth2_scheme)
        ):
            """Upload biometric data for an identity."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Read file content
            file_content = await biometric_file.read()
            
            # In a real system, this would process and store the biometric data
            # For demonstration purposes, we'll just return success
            return {
                "success": True,
                "user_id": user_id,
                "biometric_type": biometric_type,
                "file_size": len(file_content)
            }
        
        @router.post("/upload-document", response_model=Dict[str, Any])
        async def upload_document(
            user_id: str = Form(...),
            document_type: str = Form(...),
            document_file: UploadFile = File(...),
            token: str = Depends(self.oauth2_scheme)
        ):
            """Upload a document for identity verification."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Read file content
            file_content = await document_file.read()
            
            # In a real system, this would process and store the document
            # For demonstration purposes, we'll just return success
            return {
                "success": True,
                "user_id": user_id,
                "document_type": document_type,
                "file_size": len(file_content)
            }
        
        @router.get("/{user_id}", response_model=Dict[str, Any])
        async def get_identity(
            user_id: str,
            token: str = Depends(self.oauth2_scheme)
        ):
            """Retrieve identity information for a user."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Get identity
            result = self.identity_manager.get_identity(user_id, current_user)
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["error"]
                )
            
            return result
        
        @router.put("/permissions", response_model=Dict[str, Any])
        async def update_permissions(
            permissions_data: PermissionsUpdateRequest,
            token: str = Depends(self.oauth2_scheme)
        ):
            """Update access permissions for an identity."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Update permissions
            result = self.identity_manager.update_permissions(
                permissions_data.user_id,
                {
                    "authorized_viewers": permissions_data.authorized_viewers,
                    "expiration": permissions_data.expiration
                },
                current_user
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
        
        return router
    
    def _create_verification_router(self):
        """Create the identity verification router."""
        router = APIRouter()
        
        @router.post("/verify", response_model=Dict[str, Any])
        async def verify_identity(
            verification_data: VerificationRequest,
            token: str = Depends(self.oauth2_scheme)
        ):
            """Verify an identity using the specified method."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Verify identity
            result = self.identity_manager.verify_identity(
                verification_data.user_id,
                verification_data.verification_type,
                verification_data.verification_data
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
        
        @router.post("/verify-face", response_model=Dict[str, Any])
        async def verify_face(
            user_id: str = Form(...),
            reference_image: UploadFile = File(...),
            verification_image: UploadFile = File(...),
            token: str = Depends(self.oauth2_scheme)
        ):
            """Verify an identity using facial recognition."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Read files
            reference_data = await reference_image.read()
            verification_data = await verification_image.read()
            
            # Create verification data
            verification_request_data = {
                "reference_image": reference_data,
                "verification_image": verification_data,
                "ip_reputation": 0.9,  # In a real system, these would be calculated
                "geo_velocity": 0.8,
                "device_reputation": 0.95,
                "behavioral_score": 0.9
            }
            
            # Verify identity
            result = self.identity_manager.verify_identity(
                user_id,
                "facial",
                verification_request_data
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
        
        @router.post("/verify-document", response_model=Dict[str, Any])
        async def verify_document(
            user_id: str = Form(...),
            document_image: UploadFile = File(...),
            token: str = Depends(self.oauth2_scheme)
        ):
            """Verify an identity using document verification."""
            # Verify token
            current_user = self._get_current_user(token)
            
            # Read files
            document_data = await document_image.read()
            
            # Create verification data
            verification_request_data = {
                "document_image": document_data,
                "ip_reputation": 0.9,  # In a real system, these would be calculated
                "geo_velocity": 0.8,
                "device_reputation": 0.95,
                "behavioral_score": 0.9
            }
            
            # Verify identity
            result = self.identity_manager.verify_identity(
                user_id,
                "document",
                verification_request_data
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
        
        return router
    
    def _get_current_user(self, token: str):
        """
        Validate the JWT token and extract the current user.
        
        Args:
            token (str): The JWT token
            
        Returns:
            str: The current user's username
            
        Raises:
            HTTPException: If the token is invalid
        """
        try:
            # Decode the JWT token
            payload = jwt.decode(token, self.security_config["jwt_secret"], algorithms=["HS256"])
            username = payload.get("sub")
            expiration = datetime.datetime.fromtimestamp(payload.get("exp"))
            
            # Check if token is expired
            if expiration < datetime.datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return username
            
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            ) 