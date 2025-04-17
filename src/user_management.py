"""
User Management Module for V3 Watchdog AI.

Provides functionality for user authentication, role-based permissions,
and session management within the application.
"""

import os
import json
import hashlib
import hmac
import base64
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import streamlit as st

class UserRole(str, Enum):
    """User roles for permission levels."""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    """Permissions for various actions in the application."""
    VIEW_DASHBOARD = "view_dashboard"
    RUN_ANALYSIS = "run_analysis"
    UPLOAD_DATA = "upload_data"
    MODIFY_DATA = "modify_data"
    ACCESS_REPORTS = "access_reports"
    SCHEDULE_REPORTS = "schedule_reports"
    MANAGE_USERS = "manage_users"
    ACCESS_SETTINGS = "access_settings"

# Default permissions by role
DEFAULT_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.VIEW_DASHBOARD,
        Permission.RUN_ANALYSIS,
        Permission.UPLOAD_DATA,
        Permission.MODIFY_DATA,
        Permission.ACCESS_REPORTS,
        Permission.SCHEDULE_REPORTS,
        Permission.MANAGE_USERS,
        Permission.ACCESS_SETTINGS
    ],
    UserRole.MANAGER: [
        Permission.VIEW_DASHBOARD,
        Permission.RUN_ANALYSIS,
        Permission.UPLOAD_DATA,
        Permission.MODIFY_DATA,
        Permission.ACCESS_REPORTS,
        Permission.SCHEDULE_REPORTS
    ],
    UserRole.ANALYST: [
        Permission.VIEW_DASHBOARD,
        Permission.RUN_ANALYSIS,
        Permission.UPLOAD_DATA,
        Permission.ACCESS_REPORTS
    ],
    UserRole.VIEWER: [
        Permission.VIEW_DASHBOARD,
        Permission.ACCESS_REPORTS
    ]
}

class User:
    """Represents a user with authentication and permission information."""
    
    def __init__(self, username: str, password_hash: str, role: UserRole, 
                email: Optional[str] = None, 
                full_name: Optional[str] = None,
                permissions: Optional[List[Permission]] = None):
        """
        Initialize a user.
        
        Args:
            username: Unique username for the user
            password_hash: Hashed password for authentication
            role: User role for default permissions
            email: Optional email address
            full_name: Optional full name
            permissions: Optional specific permissions (overrides role defaults)
        """
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.email = email
        self.full_name = full_name
        self.last_login = None
        self.created_at = datetime.now().isoformat()
        
        # Set permissions based on role if not explicitly provided
        if permissions is None:
            self.permissions = DEFAULT_PERMISSIONS.get(role, [])
        else:
            self.permissions = permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for serialization."""
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role,
            "email": self.email,
            "full_name": self.full_name,
            "permissions": self.permissions,
            "last_login": self.last_login,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create a user from a dictionary."""
        user = cls(
            username=data["username"],
            password_hash=data["password_hash"],
            role=data["role"],
            email=data.get("email"),
            full_name=data.get("full_name"),
            permissions=data.get("permissions")
        )
        user.last_login = data.get("last_login")
        user.created_at = data.get("created_at", datetime.now().isoformat())
        return user
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if the user has a specific permission."""
        return permission in self.permissions
    
    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.now().isoformat()


class UserManager:
    """Manages users, authentication, and permissions."""
    
    def __init__(self, users_file: str = None):
        """
        Initialize the user manager.
        
        Args:
            users_file: Path to the JSON file storing user information
        """
        self.users_file = users_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                     "data", "users.json")
        self.users = {}
        self.load_users()
        
        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()
    
    def load_users(self) -> None:
        """Load users from the users file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    user_data = json.load(f)
                
                for username, data in user_data.items():
                    self.users[username] = User.from_dict(data)
                    
                print(f"Loaded {len(self.users)} users from {self.users_file}")
            except Exception as e:
                print(f"Error loading users: {e}")
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
    
    def save_users(self) -> None:
        """Save users to the users file."""
        try:
            user_data = {username: user.to_dict() for username, user in self.users.items()}
            
            with open(self.users_file, 'w') as f:
                json.dump(user_data, f, indent=2)
                
            print(f"Saved {len(self.users)} users to {self.users_file}")
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _create_default_admin(self) -> None:
        """Create a default admin user if no users exist."""
        # Create a random password for first-time setup
        admin_password = str(uuid.uuid4())[:8]  # Use first 8 chars of a UUID
        
        # Hash the password
        password_hash = self._hash_password(admin_password)
        
        # Create the admin user
        admin_user = User(
            username="admin",
            password_hash=password_hash,
            role=UserRole.ADMIN,
            email="admin@example.com",
            full_name="Administrator"
        )
        
        # Add to users dictionary
        self.users["admin"] = admin_user
        
        # Save to file
        self.save_users()
        
        print(f"Created default admin user with password: {admin_password}")
        print("Please change this password immediately after first login.")
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password for secure storage.
        
        Args:
            password: Plain text password
            
        Returns:
            Secure hash of the password
        """
        # Use a more secure method than plain SHA-256
        salt = os.urandom(32)  # 32 bytes of random data
        
        # Use PBKDF2 with HMAC-SHA256
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # Number of iterations
        )
        
        # Combine salt and key for storage
        storage = salt + key
        
        # Encode as base64 for text storage
        return base64.b64encode(storage).decode('utf-8')
    
    def verify_password(self, stored_hash: str, password: str) -> bool:
        """
        Verify a password against a stored hash.
        
        Args:
            stored_hash: The stored password hash
            password: Plain text password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Decode from base64
            storage = base64.b64decode(stored_hash.encode('utf-8'))
            
            # Extract salt (first 32 bytes) and stored key
            salt = storage[:32]
            stored_key = storage[32:]
            
            # Generate key with same process
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            # Compare in constant time to prevent timing attacks
            return hmac.compare_digest(key, stored_key)
        except Exception:
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user by username and password.
        
        Args:
            username: Username to check
            password: Password to verify
            
        Returns:
            User object if authentication succeeds, None otherwise
        """
        if username not in self.users:
            return None
        
        user = self.users[username]
        
        if self.verify_password(user.password_hash, password):
            # Update last login time
            user.update_last_login()
            self.save_users()
            return user
        
        return None
    
    def create_user(self, username: str, password: str, role: UserRole, 
                   email: Optional[str] = None, 
                   full_name: Optional[str] = None,
                   permissions: Optional[List[Permission]] = None) -> bool:
        """
        Create a new user.
        
        Args:
            username: Unique username for the user
            password: Plain text password (will be hashed)
            role: User role
            email: Optional email address
            full_name: Optional full name
            permissions: Optional specific permissions
            
        Returns:
            True if user was created, False if username already exists
        """
        if username in self.users:
            return False
        
        # Hash the password
        password_hash = self._hash_password(password)
        
        # Create the user
        user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            email=email,
            full_name=full_name,
            permissions=permissions
        )
        
        # Add to users dictionary
        self.users[username] = user
        
        # Save to file
        self.save_users()
        
        return True
    
    def update_user(self, username: str, 
                   role: Optional[UserRole] = None,
                   email: Optional[str] = None, 
                   full_name: Optional[str] = None,
                   permissions: Optional[List[Permission]] = None) -> bool:
        """
        Update an existing user.
        
        Args:
            username: Username of the user to update
            role: Optional new role
            email: Optional new email
            full_name: Optional new full name
            permissions: Optional new permissions
            
        Returns:
            True if user was updated, False if user not found
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Update fields if provided
        if role is not None:
            user.role = role
            if permissions is None:
                # Update permissions based on new role
                user.permissions = DEFAULT_PERMISSIONS.get(role, [])
        
        if email is not None:
            user.email = email
        
        if full_name is not None:
            user.full_name = full_name
        
        if permissions is not None:
            user.permissions = permissions
        
        # Save to file
        self.save_users()
        
        return True
    
    def change_password(self, username: str, password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            username: Username of the user
            password: New plain text password
            
        Returns:
            True if password was changed, False if user not found
        """
        if username not in self.users:
            return False
        
        # Hash the new password
        password_hash = self._hash_password(password)
        
        # Update the user's password hash
        self.users[username].password_hash = password_hash
        
        # Save to file
        self.save_users()
        
        return True
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user.
        
        Args:
            username: Username of the user to delete
            
        Returns:
            True if user was deleted, False if user not found
        """
        if username not in self.users:
            return False
        
        # Remove from users dictionary
        del self.users[username]
        
        # Save to file
        self.save_users()
        
        return True
    
    def get_all_users(self) -> List[User]:
        """
        Get all users.
        
        Returns:
            List of all users
        """
        return list(self.users.values())
    
    def get_user(self, username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            username: Username to look up
            
        Returns:
            User object if found, None otherwise
        """
        return self.users.get(username)


def render_login_page() -> Optional[User]:
    """
    Render the login page for user authentication.
    
    Returns:
        Authenticated user or None if not logged in
    """
    # Check if already logged in
    if 'user' in st.session_state:
        return st.session_state['user']
    
    st.title("Watchdog AI - Login")
    
    # Add styling for the login form
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the login form
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please enter both username and password.")
                return None
            
            # Initialize user manager
            user_manager = UserManager()
            
            # Authenticate the user
            user = user_manager.authenticate(username, password)
            if user:
                st.success(f"Welcome, {user.full_name or user.username}!")
                st.session_state['user'] = user
                return user
            else:
                st.error("Invalid username or password.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return None


def render_user_management() -> None:
    """Render the user management interface."""
    # Ensure only admins can access this page
    if 'user' not in st.session_state or not st.session_state['user'].has_permission(Permission.MANAGE_USERS):
        st.error("You do not have permission to access user management.")
        return
    
    st.title("User Management")
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Create tabs for different user management functions
    tab1, tab2, tab3 = st.tabs(["Users", "Create User", "Roles & Permissions"])
    
    with tab1:
        st.header("User Accounts")
        
        # Get all users
        users = user_manager.get_all_users()
        
        # Create a table of users
        user_data = []
        for user in users:
            user_data.append({
                "Username": user.username,
                "Role": user.role,
                "Email": user.email or "",
                "Name": user.full_name or "",
                "Last Login": user.last_login or "Never"
            })
        
        # Display the user table
        if user_data:
            st.dataframe(user_data, use_container_width=True)
        else:
            st.info("No users found.")
        
        # User actions section
        st.subheader("User Actions")
        
        # Select a user to manage
        selected_username = st.selectbox("Select User", [user.username for user in users])
        
        if selected_username:
            selected_user = user_manager.get_user(selected_username)
            
            if selected_user:
                # Display current user information
                st.json(selected_user.to_dict())
                
                # User management actions
                action = st.radio("Action", ["Update Information", "Change Password", "Delete User"])
                
                if action == "Update Information":
                    # Form for updating user information
                    with st.form("update_user_form"):
                        new_role = st.selectbox("Role", [role.value for role in UserRole], 
                                               index=[role.value for role in UserRole].index(selected_user.role))
                        new_email = st.text_input("Email", selected_user.email or "")
                        new_name = st.text_input("Full Name", selected_user.full_name or "")
                        
                        # Custom permissions
                        st.subheader("Permissions")
                        custom_permissions = []
                        for permission in Permission:
                            if st.checkbox(permission.value, value=permission in selected_user.permissions):
                                custom_permissions.append(permission)
                        
                        if st.form_submit_button("Update User"):
                            success = user_manager.update_user(
                                selected_username,
                                role=new_role,
                                email=new_email,
                                full_name=new_name,
                                permissions=custom_permissions
                            )
                            
                            if success:
                                st.success("User updated successfully.")
                            else:
                                st.error("Failed to update user.")
                
                elif action == "Change Password":
                    # Form for changing user password
                    with st.form("change_password_form"):
                        new_password = st.text_input("New Password", type="password")
                        confirm_password = st.text_input("Confirm Password", type="password")
                        
                        if st.form_submit_button("Change Password"):
                            if new_password != confirm_password:
                                st.error("Passwords do not match.")
                            elif not new_password:
                                st.error("Password cannot be empty.")
                            else:
                                success = user_manager.change_password(selected_username, new_password)
                                
                                if success:
                                    st.success("Password changed successfully.")
                                else:
                                    st.error("Failed to change password.")
                
                elif action == "Delete User":
                    # Confirmation for deleting user
                    st.warning(f"Are you sure you want to delete user '{selected_username}'? This action cannot be undone.")
                    
                    confirm = st.text_input("Type the username to confirm deletion")
                    
                    if st.button("Delete User"):
                        if confirm == selected_username:
                            # Prevent deleting own account
                            if selected_username == st.session_state['user'].username:
                                st.error("You cannot delete your own account.")
                            else:
                                success = user_manager.delete_user(selected_username)
                                
                                if success:
                                    st.success("User deleted successfully.")
                                else:
                                    st.error("Failed to delete user.")
                        else:
                            st.error("Username does not match. Deletion cancelled.")
    
    with tab2:
        st.header("Create New User")
        
        # Form for creating a new user
        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            new_role = st.selectbox("Role", [role.value for role in UserRole])
            new_email = st.text_input("Email")
            new_name = st.text_input("Full Name")
            
            # Custom permissions
            st.subheader("Permissions")
            st.caption("By default, permissions are based on the selected role. You can customize them below.")
            use_default_permissions = st.checkbox("Use default permissions for role", value=True)
            
            custom_permissions = []
            if not use_default_permissions:
                for permission in Permission:
                    if st.checkbox(permission.value, value=permission in DEFAULT_PERMISSIONS[new_role]):
                        custom_permissions.append(permission)
            
            if st.form_submit_button("Create User"):
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # Use default permissions if checkbox is selected
                    permissions = None if use_default_permissions else custom_permissions
                    
                    success = user_manager.create_user(
                        username=new_username,
                        password=new_password,
                        role=new_role,
                        email=new_email,
                        full_name=new_name,
                        permissions=permissions
                    )
                    
                    if success:
                        st.success(f"User '{new_username}' created successfully.")
                    else:
                        st.error(f"Failed to create user. Username '{new_username}' may already exist.")
    
    with tab3:
        st.header("Roles & Permissions")
        
        # Display information about roles and permissions
        st.markdown("""
        The system has predefined roles with different permission levels:
        
        | Role | Description |
        | --- | --- |
        | Admin | Full system access including user management |
        | Manager | Can view all data, run analysis, and schedule reports |
        | Analyst | Can upload data, run analysis, and view reports |
        | Viewer | Can only view dashboards and reports |
        """)
        
        # Display default permissions by role
        st.subheader("Default Permissions")
        
        for role in UserRole:
            with st.expander(f"{role.value} Permissions"):
                permissions = DEFAULT_PERMISSIONS.get(role, [])
                for permission in Permission:
                    st.checkbox(permission.value, value=permission in permissions, disabled=True)


def logout_user() -> None:
    """Log out the current user by clearing the session state."""
    if 'user' in st.session_state:
        del st.session_state['user']
    
    # Clear other session state data as needed
    for key in list(st.session_state.keys()):
        if key != 'page':  # Keep page state to redirect to login
            del st.session_state[key]


def render_user_menu() -> None:
    """Render a user menu in the sidebar."""
    if 'user' in st.session_state:
        user = st.session_state['user']
        
        st.sidebar.markdown(f"### Welcome, {user.full_name or user.username}")
        st.sidebar.markdown(f"**Role:** {user.role}")
        
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()
        
        # Admin link to user management
        if user.has_permission(Permission.MANAGE_USERS):
            if st.sidebar.button("User Management"):
                st.session_state['page'] = 'user_management'
                st.rerun()


def check_permission(permission: Permission) -> bool:
    """
    Check if the current user has a specific permission.
    
    Args:
        permission: The permission to check
        
    Returns:
        True if the user has permission, False otherwise
    """
    if 'user' not in st.session_state:
        return False
    
    return st.session_state['user'].has_permission(permission)


if __name__ == "__main__":
    # Example usage
    import streamlit as st
    
    st.set_page_config(page_title="User Management Demo", layout="wide")
    
    # Ensure user is logged in
    user = render_login_page()
    
    if user:
        render_user_menu()
        
        # Display a test page
        st.title("User Management Demo")
        
        st.write(f"Logged in as: {user.username}")
        st.write(f"Role: {user.role}")
        
        # Display permissions
        st.subheader("Your Permissions")
        for permission in Permission:
            st.checkbox(permission.value, value=user.has_permission(permission), disabled=True)
        
        # Check for admin permissions to show user management
        if user.has_permission(Permission.MANAGE_USERS):
            st.subheader("User Management")
            render_user_management()