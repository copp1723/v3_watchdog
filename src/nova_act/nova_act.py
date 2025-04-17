import os
import time
import logging
import streamlit as st
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta

# Import Nova components
from .scheduler import NovaScheduler
from .task import NovaTask, TaskStatus, TaskPriority
from .fallback import NovaFallback
from .credentials import NovaCredential
from .watchdog_upload import WatchdogUploader

class NovaAct:
    """
    The main Nova Act orchestration system.
    Integrates all Nova components for a complete workflow automation system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Nova Act system.
        
        Args:
            config: Configuration dictionary with settings for all components
        """
        self.logger = logging.getLogger("NovaAct")
        self.config = config or {}
        self.initialized = False
        self.start_time = datetime.now()
        
        # Set up configuration with defaults
        self._setup_config()
        
        # Initialize components
        self._init_components()
        
        # Track active operations
        self.active_operations = {}
        self.operation_history = []
        
        self.logger.info("NovaAct system initialized")
        self.initialized = True
    
    def _setup_config(self):
        """Set up configuration with defaults for missing values."""
        defaults = {
            "storage_dir": "data/nova_act",
            "credentials_path": "config/credentials.enc",
            "upload_dir": "uploads",
            "log_level": "INFO",
            "auto_retry": True,
            "max_retries": 3,
            "scheduler_interval": 5,  # seconds
            "timeout_default": 300,  # seconds
            "enable_fallback": True,
            "enable_ui": True
        }
        
        # Apply defaults for missing config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        # Create necessary directories
        for dir_path in [self.config["storage_dir"], self.config["upload_dir"]]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config["storage_dir"], "nova_act.log"))
            ]
        )
    
    def _init_components(self):
        """Initialize all Nova Act components."""
        try:
            # Initialize scheduler
            self.scheduler = NovaScheduler()
            self.scheduler.start()
            
            # Initialize credential manager
            self.credential_manager = NovaCredential(
                storage_path=self.config["credentials_path"]
            )
            
            # Initialize fallback handler
            self.fallback = NovaFallback(
                title="Nova Act Requires Attention"
            )
            
            # Initialize file uploader
            self.uploader = WatchdogUploader(
                upload_dir=self.config["upload_dir"]
            )
            
            # Register scheduled tasks
            self._register_scheduled_tasks()
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            self.logger.exception("Exception details:")
            raise
    
    def _register_scheduled_tasks(self):
        """Register default scheduled tasks."""
        # Scheduled task to check for completed operations and clean up
        self.scheduler.schedule_task(
            func=self._cleanup_operations,
            schedule_type="interval",
            interval=60 * 10,  # Every 10 minutes
            name="Operation Cleanup",
            task_id="system_operation_cleanup"
        )
    
    def create_task(self, 
                   name: str, 
                   func: Callable, 
                   args: List = None, 
                   kwargs: Dict[str, Any] = None,
                   task_type: str = "generic",
                   priority: Union[TaskPriority, int] = TaskPriority.NORMAL,
                   timeout: Optional[float] = None,
                   metadata: Dict[str, Any] = None) -> NovaTask:
        """
        Create a new task to be executed.
        
        Args:
            name: Task name
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            task_type: Type of task
            priority: Task priority
            timeout: Task timeout in seconds
            metadata: Additional metadata
            
        Returns:
            The created NovaTask
        """
        args = args or []
        kwargs = kwargs or {}
        timeout = timeout or self.config.get("timeout_default")
        
        # Create the task
        task = NovaTask(
            name=name,
            description=f"Task created by NovaAct at {datetime.now().isoformat()}",
            priority=priority,
            task_type=task_type,
            payload=metadata or {},
            timeout=timeout
        )
        
        # Set up callbacks
        task.on("on_complete", self._on_task_complete)
        task.on("on_fail", self._on_task_fail)
        
        self.logger.info(f"Created task: {task.name} ({task.task_id})")
        
        # Store in active operations
        self.active_operations[task.task_id] = {
            "type": "task",
            "task": task,
            "created_at": datetime.now(),
            "func": func,
            "args": args,
            "kwargs": kwargs
        }
        
        return task
    
    def execute_task(self, task: NovaTask) -> NovaTask:
        """
        Execute a task immediately.
        
        Args:
            task: The task to execute
            
        Returns:
            The executed task
        """
        if task.task_id not in self.active_operations:
            raise ValueError(f"Task {task.task_id} not found in active operations")
        
        operation = self.active_operations[task.task_id]
        
        self.logger.info(f"Executing task: {task.name} ({task.task_id})")
        
        # Execute the task
        task.execute(operation["func"], *operation["args"], **operation["kwargs"])
        
        return task
    
    def schedule_task(self, 
                     name: str, 
                     func: Callable, 
                     schedule_type: str,
                     args: List = None, 
                     kwargs: Dict[str, Any] = None,
                     task_type: str = "scheduled",
                     priority: Union[TaskPriority, int] = TaskPriority.NORMAL,
                     timeout: Optional[float] = None,
                     metadata: Dict[str, Any] = None,
                     **schedule_params) -> str:
        """
        Schedule a task for future execution.
        
        Args:
            name: Task name
            func: Function to execute
            schedule_type: Type of schedule ('once', 'interval', 'daily', 'weekly')
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            task_type: Type of task
            priority: Task priority
            timeout: Task timeout in seconds
            metadata: Additional metadata
            **schedule_params: Additional scheduling parameters
            
        Returns:
            Task ID of the scheduled task
        """
        args = args or []
        kwargs = kwargs or {}
        timeout = timeout or self.config.get("timeout_default")
        
        # Create a wrapper function that creates and executes a task
        def task_wrapper():
            task = self.create_task(
                name=name,
                func=func,
                args=args,
                kwargs=kwargs,
                task_type=task_type,
                priority=priority,
                timeout=timeout,
                metadata=metadata
            )
            return self.execute_task(task)
        
        # Schedule the task
        task_id = self.scheduler.schedule_task(
            func=task_wrapper,
            schedule_type=schedule_type,
            name=name,
            **schedule_params
        )
        
        self.logger.info(f"Scheduled task {name} with ID {task_id}, type: {schedule_type}")
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task or scheduled task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        # Check active operations first
        if task_id in self.active_operations:
            operation = self.active_operations[task_id]
            
            if operation["type"] == "task":
                result = operation["task"].cancel()
                if result:
                    self.logger.info(f"Cancelled task: {task_id}")
                return result
        
        # Try as a scheduled task
        result = self.scheduler.cancel_task(task_id)
        if result:
            self.logger.info(f"Cancelled scheduled task: {task_id}")
        return result
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Status information
        """
        # Check active operations
        if task_id in self.active_operations:
            operation = self.active_operations[task_id]
            
            if operation["type"] == "task":
                task = operation["task"]
                return {
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": task.status.value,
                    "progress": task.progress,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "error": task.error
                }
        
        # Check scheduler
        scheduled_task = self.scheduler.get_task_status(task_id)
        if scheduled_task:
            return {
                "task_id": task_id,
                "name": scheduled_task.get("name", "Unknown"),
                "status": "scheduled",
                "next_run": scheduled_task.get("next_run"),
                "schedule_type": scheduled_task.get("schedule_type")
            }
        
        # Check history
        for entry in self.operation_history:
            if entry.get("task_id") == task_id:
                return entry
        
        return {"task_id": task_id, "status": "not_found"}
    
    def register_fallback_action(self, 
                               title: str, 
                               description: str,
                               action_type: str,
                               retry_callback: Optional[Callable] = None,
                               skip_callback: Optional[Callable] = None,
                               details: Dict[str, Any] = None) -> str:
        """
        Register a fallback action that requires manual intervention.
        
        Args:
            title: Short title describing the issue
            description: Detailed description of what went wrong and how to fix it
            action_type: Type of action required
            retry_callback: Function to call when user wants to retry
            skip_callback: Function to call when user wants to skip
            details: Additional details about the issue
            
        Returns:
            action_id: The registered action ID
        """
        # Generate a unique action ID
        action_id = f"action_{int(time.time())}_{len(self.fallback.get_all_pending_actions())}"
        
        # Register with the fallback handler
        self.fallback.register_fallback_action(
            action_id=action_id,
            title=title,
            description=description,
            action_type=action_type,
            retry_callback=retry_callback,
            skip_callback=skip_callback,
            details=details
        )
        
        self.logger.info(f"Registered fallback action: {title} ({action_id})")
        
        return action_id
    
    def verify_credentials(self, system_id: str, test_func: Callable = None) -> Dict[str, Any]:
        """
        Verify credentials for a system.
        
        Args:
            system_id: Identifier for the system
            test_func: Function to test the credentials
            
        Returns:
            Verification results
        """
        return self.credential_manager.verify_credential(system_id, test_func)
    
    def add_credentials(self, 
                      system_id: str, 
                      username: str, 
                      password: str, 
                      metadata: Dict[str, Any] = None) -> bool:
        """
        Add credentials for a system.
        
        Args:
            system_id: Identifier for the system
            username: Username
            password: Password
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        return self.credential_manager.add_credential(
            system_id=system_id,
            username=username,
            password=password,
            metadata=metadata
        )
    
    def get_credentials(self, system_id: str) -> Optional[Dict[str, Any]]:
        """
        Get credentials for a system.
        
        Args:
            system_id: Identifier for the system
            
        Returns:
            Credential information or None if not found
        """
        return self.credential_manager.get_credential(system_id)
    
    def render_upload_ui(self, **kwargs) -> Dict:
        """
        Render the file upload UI component.
        
        Args:
            **kwargs: Arguments to pass to the uploader
            
        Returns:
            Upload result information
        """
        return self.uploader.render_upload_ui(**kwargs)
    
    def render_fallback_ui(self):
        """Render the fallback UI for manual interventions."""
        if self.fallback.has_pending_actions():
            self.fallback.render_fallback_ui()
    
    def render_status_ui(self):
        """Render a status dashboard for Nova Act operations."""
        st.subheader("Nova Act System Status")
        
        # System uptime
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        st.markdown(f"**System Uptime:** {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Active operations
        st.markdown(f"**Active Operations:** {len(self.active_operations)}")
        
        # Scheduled tasks
        scheduled_tasks = self.scheduler.get_all_tasks()
        st.markdown(f"**Scheduled Tasks:** {len(scheduled_tasks)}")
        
        # Pending fallback actions
        pending_actions = self.fallback.get_all_pending_actions()
        st.markdown(f"**Pending Manual Actions:** {len(pending_actions)}")
        
        if pending_actions:
            st.warning(f"{len(pending_actions)} issues require manual intervention")
        
        # Recent activity
        if self.operation_history:
            st.markdown("### Recent Activity")
            
            for entry in sorted(self.operation_history[-10:], 
                               key=lambda x: x.get("timestamp", ""), reverse=True):
                status = entry.get("status", "unknown")
                icon = "✅" if status == "completed" else "❌" if status in ["failed", "error"] else "⏱️"
                
                st.markdown(f"{icon} **{entry.get('name', 'Unknown operation')}** - "
                           f"{entry.get('status', 'unknown')} "
                           f"({entry.get('timestamp', '')})")
    
    def _on_task_complete(self, task: NovaTask):
        """Callback for when a task completes successfully."""
        self.logger.info(f"Task completed: {task.name} ({task.task_id})")
        
        # Add to history
        self.operation_history.append({
            "task_id": task.task_id,
            "name": task.name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "result": task.result,
            "execution_time": task._execution_time
        })
    
    def _on_task_fail(self, task: NovaTask):
        """Callback for when a task fails."""
        self.logger.warning(f"Task failed: {task.name} ({task.task_id}) - {task.error}")
        
        # Add to history
        self.operation_history.append({
            "task_id": task.task_id,
            "name": task.name,
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": task.error,
            "execution_time": task._execution_time
        })
        
        # Auto-retry if enabled and retries left
        if (self.config.get("auto_retry", True) and 
            task.retry_count < task.max_retries and
            task.status != TaskStatus.CANCELLED):
            
            operation = self.active_operations.get(task.task_id)
            if operation:
                self.logger.info(f"Auto-retrying task: {task.name} (attempt {task.retry_count}/{task.max_retries})")
                
                # Reset task status to pending
                task._set_status(TaskStatus.PENDING)
                
                # Execute again
                task.execute(operation["func"], *operation["args"], **operation["kwargs"])
    
    def _cleanup_operations(self):
        """Clean up completed and failed operations after a certain time."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for task_id, operation in self.active_operations.items():
            if operation["type"] == "task":
                task = operation["task"]
                
                # If task is completed or failed and older than 1 hour
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.last_updated and
                    current_time - task.last_updated > timedelta(hours=1)):
                    
                    keys_to_remove.append(task_id)
        
        # Remove old operations
        for task_id in keys_to_remove:
            del self.active_operations[task_id]
        
        # Trim history if it gets too large
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]  # Keep most recent 500
        
        self.logger.debug(f"Cleaned up {len(keys_to_remove)} completed operations")
        
        return {
            "cleaned_count": len(keys_to_remove),
            "remaining_count": len(self.active_operations),
            "history_size": len(self.operation_history)
        }
    
    def shutdown(self):
        """Shut down the Nova Act system gracefully."""
        self.logger.info("Shutting down Nova Act system...")
        
        # Stop the scheduler
        self.scheduler.stop()
        
        # Save any pending changes
        self.credential_manager.save()
        
        self.logger.info("Nova Act system shutdown complete")
        
        return True 