import uuid
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum

class TaskStatus(Enum):
    """Enum for task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Enum for task priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class NovaTask:
    """
    Represents a task in the Nova Act system.
    Tasks are units of work that can be executed, tracked, and managed.
    """
    
    def __init__(self, 
                task_id: Optional[str] = None,
                name: str = "Unnamed Task",
                description: str = "",
                priority: Union[TaskPriority, int] = TaskPriority.NORMAL,
                task_type: str = "generic",
                payload: Dict[str, Any] = None,
                timeout: Optional[float] = None):
        """
        Initialize a new task.
        
        Args:
            task_id: Unique ID for this task (generated if None)
            name: Human-readable name for the task
            description: Detailed description of the task
            priority: Task priority level
            task_type: Category/type of task
            payload: Data associated with this task
            timeout: Maximum execution time in seconds (None for no timeout)
        """
        self.task_id = task_id if task_id else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.priority = priority if isinstance(priority, TaskPriority) else TaskPriority(priority)
        self.task_type = task_type
        self.payload = payload or {}
        self.timeout = timeout
        
        # Task execution metadata
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.last_updated = self.created_at
        self.result = None
        self.error = None
        self.progress = 0.0
        self.retry_count = 0
        self.max_retries = 3
        self.dependencies = []
        self.subtasks = []
        self.parent_id = None
        self.log_entries = []
        self.tags = []
        
        # Internal trackers
        self._execution_time = 0
        self._callbacks = {
            "on_start": [],
            "on_progress": [],
            "on_complete": [],
            "on_fail": [],
            "on_cancel": []
        }
        
        # Set up logging
        self.logger = logging.getLogger(f"NovaTask.{self.task_id}")
        self.log(f"Task created: {self.name}")
    
    def execute(self, func: Callable, *args, **kwargs) -> "NovaTask":
        """
        Execute the task by running the provided function.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            self: For method chaining
        """
        if self.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
            self.log(f"Task already {self.status.value}, cannot execute again")
            return self
            
        if self.status == TaskStatus.CANCELLED:
            self.log("Cannot execute cancelled task")
            return self
            
        # Mark as running and track start time
        self._set_status(TaskStatus.RUNNING)
        self.started_at = datetime.now()
        self._fire_callbacks("on_start", self)
        self.log(f"Starting execution of task: {self.name}")
        
        start_time = time.time()
        
        try:
            # Execute with timeout if specified
            if self.timeout:
                # Execute with timeout handling
                import threading
                result_container = []
                error_container = []
                
                def _run_with_timeout():
                    try:
                        result = func(*args, **kwargs)
                        result_container.append(result)
                    except Exception as e:
                        error_container.append(e)
                
                thread = threading.Thread(target=_run_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(self.timeout)
                
                if thread.is_alive():
                    # Timeout occurred
                    self._set_status(TaskStatus.TIMEOUT)
                    self.error = f"Task timed out after {self.timeout} seconds"
                    self._fire_callbacks("on_fail", self)
                    self.log(f"Task timed out: {self.error}", level="ERROR")
                else:
                    # Thread completed
                    if error_container:
                        raise error_container[0]
                    if result_container:
                        self.result = result_container[0]
                        self._complete_task()
                    else:
                        self._set_status(TaskStatus.FAILED)
                        self.error = "Task completed but no result captured"
                        self._fire_callbacks("on_fail", self)
            else:
                # Execute normally without timeout
                self.result = func(*args, **kwargs)
                self._complete_task()
                
        except Exception as e:
            self._set_status(TaskStatus.FAILED)
            self.error = str(e)
            self._execution_time = time.time() - start_time
            self._fire_callbacks("on_fail", self)
            self.log(f"Task failed: {str(e)}", level="ERROR")
            # Log the full exception with traceback
            self.logger.exception("Exception details:")
            
            # Check if we should retry
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self.log(f"Retrying task (attempt {self.retry_count}/{self.max_retries})")
                # Reset status to pending for the retry
                self._set_status(TaskStatus.PENDING)
                # We don't actually retry here - that's handled by the task manager
            
        return self
    
    def cancel(self) -> bool:
        """
        Cancel the task if it hasn't started running.
        
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        if self.status == TaskStatus.PENDING:
            self._set_status(TaskStatus.CANCELLED)
            self._fire_callbacks("on_cancel", self)
            self.log("Task cancelled")
            return True
        else:
            self.log(f"Cannot cancel task in {self.status.value} state")
            return False
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> "NovaTask":
        """
        Update the task's progress (0.0 to 1.0).
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional message to log with the progress update
            
        Returns:
            self: For method chaining
        """
        self.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        self.last_updated = datetime.now()
        
        if message:
            self.log(f"Progress {self.progress:.0%}: {message}")
        else:
            self.log(f"Progress updated: {self.progress:.0%}")
            
        self._fire_callbacks("on_progress", self)
        return self
    
    def add_dependency(self, task: Union[str, "NovaTask"]) -> "NovaTask":
        """
        Add a dependency that must complete before this task can run.
        
        Args:
            task: Task ID or NovaTask object that this task depends on
            
        Returns:
            self: For method chaining
        """
        task_id = task.task_id if isinstance(task, NovaTask) else task
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            self.log(f"Added dependency: {task_id}")
        return self
    
    def add_subtask(self, task: "NovaTask") -> "NovaTask":
        """
        Add a subtask to this task.
        
        Args:
            task: NovaTask object to add as a subtask
            
        Returns:
            self: For method chaining
        """
        task.parent_id = self.task_id
        self.subtasks.append(task.task_id)
        self.log(f"Added subtask: {task.task_id} ({task.name})")
        return self
    
    def add_tag(self, tag: str) -> "NovaTask":
        """
        Add a tag to this task.
        
        Args:
            tag: Tag string to add
            
        Returns:
            self: For method chaining
        """
        if tag not in self.tags:
            self.tags.append(tag)
        return self
    
    def on(self, event: str, callback: Callable[["NovaTask"], None]) -> "NovaTask":
        """
        Register a callback for a task event.
        
        Args:
            event: Event name ('on_start', 'on_progress', 'on_complete', 'on_fail', 'on_cancel')
            callback: Function to call when the event occurs
            
        Returns:
            self: For method chaining
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        return self
    
    def log(self, message: str, level: str = "INFO") -> "NovaTask":
        """
        Add a log entry to this task's log.
        
        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            self: For method chaining
        """
        timestamp = datetime.now()
        entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        self.log_entries.append(entry)
        
        # Also log to the Python logger
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        return self
    
    def set_metadata(self, key: str, value: Any) -> "NovaTask":
        """
        Set a metadata value in the task payload.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            self: For method chaining
        """
        self.payload[key] = value
        return self
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value from the task payload.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Value of the metadata key
        """
        return self.payload.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "task_type": self.task_type,
            "payload": self.payload,
            "timeout": self.timeout,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "execution_time": self._execution_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NovaTask":
        """
        Create a task from a dictionary (deserialization).
        
        Args:
            data: Dictionary containing task data
            
        Returns:
            NovaTask instance
        """
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", "Unnamed Task"),
            description=data.get("description", ""),
            priority=data.get("priority", TaskPriority.NORMAL.value),
            task_type=data.get("task_type", "generic"),
            payload=data.get("payload", {}),
            timeout=data.get("timeout")
        )
        
        # Restore status
        status_value = data.get("status", "pending")
        task.status = TaskStatus(status_value)
        
        # Restore timestamps
        for attr in ["created_at", "started_at", "completed_at", "last_updated"]:
            if data.get(attr):
                try:
                    setattr(task, attr, datetime.fromisoformat(data[attr]))
                except (ValueError, TypeError):
                    # Handle invalid datetime format
                    pass
        
        # Restore other fields
        task.result = data.get("result")
        task.error = data.get("error")
        task.progress = data.get("progress", 0.0)
        task.retry_count = data.get("retry_count", 0)
        task.max_retries = data.get("max_retries", 3)
        task.dependencies = data.get("dependencies", [])
        task.subtasks = data.get("subtasks", [])
        task.parent_id = data.get("parent_id")
        task.tags = data.get("tags", [])
        task._execution_time = data.get("execution_time", 0)
        
        return task
    
    def _set_status(self, status: TaskStatus) -> None:
        """Update the task status and last_updated timestamp."""
        self.status = status
        self.last_updated = datetime.now()
    
    def _complete_task(self) -> None:
        """Mark the task as completed and handle callbacks."""
        self._set_status(TaskStatus.COMPLETED)
        self.completed_at = datetime.now()
        self.progress = 1.0
        
        # Calculate execution time
        if self.started_at:
            self._execution_time = (self.completed_at - self.started_at).total_seconds()
        
        self._fire_callbacks("on_complete", self)
        self.log(f"Task completed in {self._execution_time:.2f}s")
    
    def _fire_callbacks(self, event: str, task: "NovaTask") -> None:
        """Fire all callbacks registered for the given event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(task)
            except Exception as e:
                self.log(f"Error in {event} callback: {str(e)}", "ERROR")
                self.logger.exception(f"Exception in {event} callback:")
                
    def __str__(self) -> str:
        """String representation of the task."""
        return f"NovaTask({self.task_id}: {self.name} - {self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of the task."""
        return (f"NovaTask(id='{self.task_id}', name='{self.name}', "
                f"type='{self.task_type}', status='{self.status.value}', "
                f"progress={self.progress:.1%})") 