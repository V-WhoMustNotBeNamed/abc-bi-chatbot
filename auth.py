"""
Authentication module for Streamlit BI application.
Implements Google OAuth 2.0 with user allowlist and role-based permissions.
"""

import streamlit as st
from google.auth.transport import requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import os
import json
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_USERS = {
    "42expelliarmus@gmail.com": {
        "role": "admin",
        "permissions": ["export", "full_access", "view_analytics", "modify_settings"]
    },
    "mx.anderson15@gmail.com": {
        "role": "staff", 
        "permissions": ["basic_access", "view_analytics"]
    }
}

# FUTURE: Domain-based restriction for production clients
# Uncomment and modify for client-specific domain restrictions
# ALLOWED_DOMAINS = ['@conceptionnurseries.com']
# def check_domain_restriction(email: str) -> bool:
#     """Check if email belongs to allowed domain."""
#     for domain in ALLOWED_DOMAINS:
#         if email.endswith(domain):
#             return True
#     return False

class GoogleAuthenticator:
    """Handles Google OAuth 2.0 authentication flow."""
    
    def __init__(self):
        """Initialize the Google authenticator with credentials from secrets."""
        self.client_id = None
        self.client_secret = None
        self.redirect_uri = None
        self._load_credentials()
    
    def _load_credentials(self):
        """Load OAuth credentials from Streamlit secrets."""
        try:
            if "google_oauth" in st.secrets:
                self.client_id = st.secrets["google_oauth"]["client_id"]
                self.client_secret = st.secrets["google_oauth"]["client_secret"]
                self.redirect_uri = st.secrets["google_oauth"].get(
                    "redirect_uri", 
                    "http://localhost:8501"
                )
            else:
                logger.warning("Google OAuth credentials not found in secrets")
        except Exception as e:
            logger.error(f"Error loading OAuth credentials: {e}")
    
    def create_flow(self) -> Optional[Flow]:
        """Create OAuth flow for Google authentication."""
        if not self.client_id or not self.client_secret:
            return None
        
        client_config = {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=[
                "openid",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile"
            ],
            redirect_uri=self.redirect_uri
        )
        
        return flow
    
    def get_authorization_url(self, flow: Flow) -> str:
        """Generate authorization URL for OAuth flow."""
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="select_account"
        )
        return authorization_url, state

def init_session_state():
    """Initialize authentication-related session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "user_permissions" not in st.session_state:
        st.session_state.user_permissions = []
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None

def check_user_authorization(email: str) -> Dict[str, any]:
    """
    Check if user is authorized and return their permissions.
    
    Args:
        email: User's email address
        
    Returns:
        Dictionary with authorization status and user details
    """
    result = {
        "authorized": False,
        "role": None,
        "permissions": [],
        "message": ""
    }
    
    # Check against allowlist
    if email in ALLOWED_USERS:
        result["authorized"] = True
        result["role"] = ALLOWED_USERS[email]["role"]
        result["permissions"] = ALLOWED_USERS[email]["permissions"]
        result["message"] = f"Welcome {email}"
    else:
        result["message"] = f"Access denied. User {email} is not authorized to access this application."
    
    # FUTURE: Uncomment for domain-based restriction
    # if not result["authorized"] and check_domain_restriction(email):
    #     result["authorized"] = True
    #     result["role"] = "staff"  # Default role for domain users
    #     result["permissions"] = ["basic_access"]
    #     result["message"] = f"Welcome {email} (Domain Access)"
    
    return result

def has_permission(permission: str) -> bool:
    """
    Check if current user has specific permission.
    
    Args:
        permission: Permission to check
        
    Returns:
        True if user has permission, False otherwise
    """
    if not st.session_state.get("authenticated", False):
        return False
    
    user_permissions = st.session_state.get("user_permissions", [])
    return permission in user_permissions or "full_access" in user_permissions

def login_user(email: str, name: str = None):
    """
    Log in a user after successful authentication.
    
    Args:
        email: User's email address
        name: User's display name (optional)
    """
    auth_result = check_user_authorization(email)
    
    if auth_result["authorized"]:
        st.session_state.authenticated = True
        st.session_state.user_email = email
        st.session_state.user_name = name or email.split("@")[0]
        st.session_state.user_role = auth_result["role"]
        st.session_state.user_permissions = auth_result["permissions"]
        logger.info(f"User {email} logged in successfully with role: {auth_result['role']}")
        return True
    else:
        logger.warning(f"Unauthorized login attempt by {email}")
        st.error(auth_result["message"])
        return False

def logout_user():
    """Log out the current user and clear session state."""
    for key in ["authenticated", "user_email", "user_name", "user_role", 
                "user_permissions", "auth_token"]:
        if key in st.session_state:
            del st.session_state[key]
    
    logger.info("User logged out successfully")
    st.rerun()

def display_login_page():
    """Display the login page with Google OAuth option."""
    st.title("🔐 AI BI Analyst - Login")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome to AI BI Analyst")
        st.markdown("Please sign in with your authorized Google account to continue.")
        
        # For development/testing: Allow bypass with hardcoded users
        if st.secrets.get("dev_mode", False):
            st.warning("⚠️ Development Mode Active")
            if st.button("Login as Admin (42expelliarmus@gmail.com)", type="primary"):
                if login_user("42expelliarmus@gmail.com", "Admin User"):
                    st.rerun()
            
            if st.button("Login as Staff (mx.anderson15@gmail.com)"):
                if login_user("mx.anderson15@gmail.com", "Staff User"):
                    st.rerun()
        else:
            # Production mode: Use actual Google OAuth
            authenticator = GoogleAuthenticator()
            
            if authenticator.client_id:
                flow = authenticator.create_flow()
                if flow:
                    auth_url, state = authenticator.get_authorization_url(flow)
                    
                    st.markdown("### Sign in with Google")
                    st.markdown(f"[🔑 Click here to sign in with Google]({auth_url})")
                    
                    # Handle OAuth callback
                    query_params = st.query_params
                    if "code" in query_params:
                        try:
                            flow.fetch_token(code=query_params["code"])
                            credentials = flow.credentials
                            
                            # Verify the token
                            request = requests.Request()
                            id_info = id_token.verify_oauth2_token(
                                credentials.id_token, 
                                request, 
                                authenticator.client_id
                            )
                            
                            email = id_info.get("email")
                            name = id_info.get("name")
                            
                            if email:
                                if login_user(email, name):
                                    st.query_params.clear()
                                    st.rerun()
                            else:
                                st.error("Could not retrieve email from Google account")
                                
                        except Exception as e:
                            st.error(f"Authentication failed: {str(e)}")
                            logger.error(f"OAuth error: {e}")
            else:
                st.error("Google OAuth is not configured. Please contact your administrator.")
                st.info("To configure OAuth, add credentials to `.streamlit/secrets.toml`")
        
        st.markdown("---")
        st.markdown("##### Authorized Users Only")
        st.markdown("This application is restricted to authorized users. If you believe you should have access, please contact your administrator.")

def display_user_info():
    """Display current user information in the sidebar."""
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.markdown("### User Information")
            st.markdown(f"**User:** {st.session_state.get('user_name', 'Unknown')}")
            st.markdown(f"**Email:** {st.session_state.get('user_email', 'Unknown')}")
            st.markdown(f"**Role:** {st.session_state.get('user_role', 'Unknown')}")
            
            # Display permissions as badges
            permissions = st.session_state.get('user_permissions', [])
            if permissions:
                st.markdown("**Permissions:**")
                perm_cols = st.columns(2)
                for i, perm in enumerate(permissions):
                    with perm_cols[i % 2]:
                        st.caption(f"✓ {perm.replace('_', ' ').title()}")
            
            st.markdown("---")
            
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                logout_user()

def require_auth(func):
    """
    Decorator to require authentication for a function.
    
    Usage:
        @require_auth
        def protected_function():
            # Function code here
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            st.error("🔒 Authentication required to access this feature")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def require_permission(permission: str):
    """
    Decorator to require specific permission for a function.
    
    Usage:
        @require_permission("export")
        def export_function():
            # Function code here
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not has_permission(permission):
                st.error(f"🚫 You don't have permission to access this feature. Required: {permission}")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_authentication():
    """
    Main authentication check function to be called at app startup.
    Returns True if authenticated, False otherwise.
    """
    init_session_state()
    
    if not st.session_state.get("authenticated", False):
        display_login_page()
        return False
    
    return True