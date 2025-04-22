"""
Sandboxed Agent Execution for Watchdog AI.

This module provides functionality to execute LLM-generated code in a secure sandbox environment.
It enforces schema validation on the code's output and provides error handling and retry mechanisms.
"""

import os
import sys
import json
import time
import logging
import traceback
import subprocess
import threading
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from functools import wraps
import pandas as pd
import numpy as np
import jsonschema
from dataclasses import dataclass
import uuid
import contextlib
from io import StringIO

# Import user-defined modules
from .log_utils_config import get_logger
from .errors import ValidationError

# Configure logger
logger = get_logger(__name__)

# Default JSON schema for agent output
DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["answer", "data", "chart_type", "confidence"],
    "properties": {
        "answer": {
            "type": "string",
            "description": "Textual answer to the user's question"
        },
        "data": {
            "type": ["object", "array"],
            "description": "JSON-serializable data for visualization"
        },
        "chart_type": {
            "type": "string",
            "enum": ["table", "bar", "line", "pie", "scatter", "none"],
            "description": "Type of chart to visualize the data"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence level in the answer (0.0 to 1.0)"
        },
        "metadata": {
            "type": "object",
            "description": "Optional metadata about the execution"
        }
    }
}

class SchemaValidationError(Exception):
    """Exception for schema validation errors in agent output."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize a schema validation error."""
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message}: {json.dumps(self.details, indent=2)}"
        return self.message

class SandboxExecutionError(Exception):
    """Exception for errors during sandbox execution."""
    
    def __init__(self, message: str, error_type: str, original_error: Optional[Exception] = None):
        """Initialize a sandbox execution error."""
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        result = f"[{self.error_type}] {self.message}"
        if self.original_error:
            result += f"\nOriginal error: {str(self.original_error)}"
        return result

@dataclass
class SandboxResult:
    """Result of a sandbox execution."""
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    code: Optional[str] = None
    log_output: Optional[str] = None

@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    memory_limit_mb: int = 512
    execution_timeout_seconds: int = 10
    enable_network: bool = False
    allowed_modules: List[str] = None  # None means use default restricted modules
    
    def __post_init__(self):
        """Set default values for allowed modules if not specified."""
        if self.allowed_modules is None:
            self.allowed_modules = [
                "pandas", "numpy", "math", "datetime", "collections", 
                "json", "re", "functools", "itertools"
            ]

def _prepare_sandbox_env() -> Dict[str, Any]:
    """
    Prepare a restricted environment for code execution.
    
    Returns:
        Dictionary with allowed environment variables and modules
    """
    # Create a restricted globals dict
    sandbox_globals = {
        "__builtins__": {
            # Allow basic types and functions
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "len": len, "range": range, "enumerate": enumerate,
            "sum": sum, "min": min, "max": max, "round": round,
            "sorted": sorted, "abs": abs, "all": all, "any": any,
            "zip": zip, "map": map, "filter": filter,
            "True": True, "False": False, "None": None,
            # Basic exceptions
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError
        },
        # Add standard modules
        "pd": pd,
        "np": np,
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "collections": __import__("collections"),
        "json": json,
        "re": __import__("re"),
        "functools": __import__("functools"),
        "itertools": __import__("itertools"),
    }
    
    return sandbox_globals

def _timeout_handler(proc):
    """Terminate a process after timeout."""
    proc.kill()

def create_safe_code_wrapper(code: str, dataframe_var: str = "df") -> str:
    """
    Wrap user code in a safe execution function.
    
    Args:
        code: The Python code to wrap
        dataframe_var: The variable name for the DataFrame in the code
        
    Returns:
        Wrapped code as a string
    """
    # Indent the code
    indented_code = "\n".join(f"    {line}" for line in code.split("\n"))
    
    # Create wrapper function
    wrapped_code = f"""
def safe_execute(df):
    # User-defined code starts here
{indented_code}
    # User-defined code ends here
    
    # Return result if not explicitly returned in the code
    if 'result' in locals():
        return result
    elif 'answer' in locals():
        # Check if we have a structured output
        if isinstance(answer, dict) and 'data' in answer and 'chart_type' in answer:
            return answer
        else:
            # Create a structured output
            return {{
                "answer": str(answer),
                "data": {{}},
                "chart_type": "none",
                "confidence": 0.8
            }}
    else:
        # Create a default output with any detected tabular data
        for var_name, var_value in locals().items():
            if isinstance(var_value, pd.DataFrame) and var_name != '{dataframe_var}':
                return {{
                    "answer": "Analysis complete. See the table below for results.",
                    "data": var_value.to_dict(orient='records'),
                    "chart_type": "table",
                    "confidence": 0.7
                }}
        
        # Fallback if no clear output was found
        return {{
            "answer": "Analysis complete, but no structured output was generated.",
            "data": {{}},
            "chart_type": "none",
            "confidence": 0.5
        }}

# Execute the safe function with provided DataFrame
result = safe_execute(df)
print("RESULT_JSON_START")
print(json.dumps(result))
print("RESULT_JSON_END")
"""
    return wrapped_code

def execute_in_subprocess(code: str, df: pd.DataFrame, config: SandboxConfig) -> SandboxResult:
    """
    Execute code in a subprocess with resource limits.
    
    Args:
        code: The Python code to execute
        df: The pandas DataFrame to provide to the code
        config: The sandbox configuration
        
    Returns:
        SandboxResult with the execution outcome
    """
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as code_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as data_file:
        
        # Save DataFrame to CSV
        df.to_csv(data_file.name, index=False)
        
        # Create the full code with imports and DataFrame loading
        full_code = f"""
import json
import sys
import os
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import collections
import re
import functools
import itertools

# Set resource limits
import resource
resource.setrlimit(resource.RLIMIT_AS, (
    {config.memory_limit_mb * 1024 * 1024},  # Soft limit
    {config.memory_limit_mb * 1024 * 1024}   # Hard limit
))

# Disable potentially unsafe modules
sys.modules.pop('subprocess', None)
sys.modules.pop('os', None)
sys.modules.pop('sys', None)
sys.modules.pop('importlib', None)
sys.modules.pop('builtins', None)

# Load the DataFrame
df = pd.read_csv('{data_file.name}')

{create_safe_code_wrapper(code)}
"""
        # Write code to the temporary file
        code_file.write(full_code.encode('utf-8'))
        code_file.flush()
    
    try:
        start_time = time.time()
        
        # Execute the code in a subprocess
        process = subprocess.Popen(
            [sys.executable, code_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Set up timeout
        timer = threading.Timer(config.execution_timeout_seconds, _timeout_handler, [process])
        timer.start()
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Cancel timer if process completed
        timer.cancel()
        
        execution_time = time.time() - start_time
        
        # Check for errors
        if process.returncode != 0:
            error_msg = stderr.strip()
            return SandboxResult(
                success=False,
                error=SandboxExecutionError(error_msg, "execution_error"),
                execution_time=execution_time,
                code=code,
                log_output=f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )
        
        # Extract the result JSON from stdout
        try:
            output_start = stdout.find("RESULT_JSON_START")
            output_end = stdout.find("RESULT_JSON_END")
            
            if output_start == -1 or output_end == -1:
                return SandboxResult(
                    success=False,
                    error=SandboxExecutionError(
                        "Couldn't find result JSON markers in output",
                        "output_format_error"
                    ),
                    execution_time=execution_time,
                    code=code,
                    log_output=f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                )
            
            json_str = stdout[output_start + len("RESULT_JSON_START"):output_end].strip()
            result = json.loads(json_str)
            
            return SandboxResult(
                success=True,
                output=result,
                execution_time=execution_time,
                code=code,
                log_output=f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )
            
        except json.JSONDecodeError as e:
            return SandboxResult(
                success=False,
                error=SandboxExecutionError(
                    f"Failed to parse JSON result: {str(e)}",
                    "json_decode_error",
                    e
                ),
                execution_time=execution_time,
                code=code,
                log_output=f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )
        
    except Exception as e:
        return SandboxResult(
            success=False,
            error=SandboxExecutionError(
                f"Unexpected error during execution: {str(e)}",
                "system_error",
                e
            ),
            execution_time=0.0,
            code=code,
            log_output=f"Exception: {traceback.format_exc()}"
        )
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(code_file.name)
            os.unlink(data_file.name)
        except:
            pass

def validate_output_against_schema(output: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate the output against the provided JSON schema.
    
    Args:
        output: The output to validate
        schema: The JSON schema to validate against
        
    Returns:
        Tuple containing:
        - success: Boolean indicating if validation passed
        - errors: Dictionary of validation errors if validation failed, None otherwise
    """
    try:
        jsonschema.validate(instance=output, schema=schema)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, {
            "message": str(e),
            "path": list(e.path),
            "schema_path": list(e.schema_path),
            "expected": e.validator_value,
            "found": e.instance
        }

def execute_code_in_sandbox(
    code: str,
    df: pd.DataFrame,
    schema: Dict[str, Any] = None,
    config: SandboxConfig = None
) -> Dict[str, Any]:
    """
    Execute code in a sandbox environment and validate the output.
    
    Args:
        code: The Python code to execute
        df: The pandas DataFrame to provide to the code
        schema: The JSON schema to validate the output against
        config: Configuration for the sandbox execution
        
    Returns:
        Dictionary containing the validated output
        
    Raises:
        SandboxExecutionError: If execution fails
        SchemaValidationError: If output validation fails
    """
    # Use default schema and config if not provided
    schema = schema or DEFAULT_OUTPUT_SCHEMA
    config = config or SandboxConfig()
    
    # Log the execution
    exec_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting sandboxed execution [{exec_id}] with timeout={config.execution_timeout_seconds}s")
    
    # Execute the code
    result = execute_in_subprocess(code, df, config)
    
    # Log the result
    if result.success:
        logger.info(f"Execution [{exec_id}] completed successfully in {result.execution_time:.2f}s")
    else:
        logger.error(f"Execution [{exec_id}] failed: {str(result.error)}")
        logger.debug(f"Execution log [{exec_id}]:\n{result.log_output}")
        raise result.error
    
    # Validate the output
    is_valid, validation_errors = validate_output_against_schema(result.output, schema)
    
    if not is_valid:
        logger.error(f"Schema validation failed for execution [{exec_id}]: {validation_errors}")
        raise SchemaValidationError(
            f"Output doesn't match the required schema",
            validation_errors
        )
    
    # Add execution metadata to the output
    if "metadata" not in result.output:
        result.output["metadata"] = {}
    
    result.output["metadata"]["execution_id"] = exec_id
    result.output["metadata"]["execution_time"] = result.execution_time
    
    return result.output

def retry_with_modified_prompt(
    execution_func: Callable,
    original_error: Exception,
    df: pd.DataFrame,
    schema: Dict[str, Any],
    llm_service_func: Callable,
    original_prompt: str,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Retry execution with a modified prompt if initial execution fails.
    
    Args:
        execution_func: Function to execute the code
        original_error: The error that triggered the retry
        df: The DataFrame to use
        schema: The output schema to validate against
        llm_service_func: Function to call the LLM service with a modified prompt
        original_prompt: The original prompt that generated the failing code
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing the validated output
        
    Raises:
        SandboxExecutionError: If all retries fail
    """
    error_message = str(original_error)
    error_type = original_error.__class__.__name__
    
    # Log the retry attempt
    logger.info(f"Attempting retry with modified prompt after {error_type}: {error_message}")
    
    # Create error context for the retry prompt
    error_context = {
        "error_type": error_type,
        "error_message": error_message,
        "schema": schema,
        "column_info": {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_values": {col: str(df[col].iloc[0]) if len(df) > 0 else None for col in df.columns}
        }
    }
    
    # Create a retry prompt
    retry_prompt = f"""
Your previous code generated for the query "{original_prompt}" failed with the following error:

Error Type: {error_type}
Error Message: {error_message}

Please fix the code to handle this error. The code must return a result that conforms to this JSON schema:
{json.dumps(schema, indent=2)}

DataFrame information:
- Columns: {list(df.columns)}
- Data types: {json.dumps({col: str(df[col].dtype) for col in df.columns}, indent=2)}
- Sample row: {df.iloc[0].to_dict() if len(df) > 0 else "No data"}

Generate ONLY the fixed Python code without explanations.
"""
    
    # Try multiple times with modified prompts
    for retry_attempt in range(max_retries):
        try:
            # Generate new code with the retry prompt
            new_code = llm_service_func(retry_prompt)
            
            # Execute the new code
            retry_result = execution_func(new_code, df, schema)
            
            # Log successful retry
            logger.info(f"Retry #{retry_attempt+1} successful after {error_type} error")
            
            # Add retry metadata
            if "metadata" not in retry_result:
                retry_result["metadata"] = {}
            
            retry_result["metadata"]["retry_attempt"] = retry_attempt + 1
            retry_result["metadata"]["original_error"] = error_message
            retry_result["metadata"]["retry_prompt"] = retry_prompt
            
            return retry_result
            
        except Exception as retry_error:
            logger.warning(f"Retry #{retry_attempt+1} failed: {str(retry_error)}")
            
            # Make the prompt more specific for the next retry if there are more attempts
            if retry_attempt < max_retries - 1:
                retry_prompt += f"\n\nThe fix still didn't work. New error: {str(retry_error)}\n"
                retry_prompt += "Please generate a simpler, more robust solution that focuses on the core task and handles edge cases."
    
    # If all retries failed, raise the original error
    logger.error(f"All {max_retries} retry attempts failed")
    raise original_error

def log_execution(result: Dict[str, Any], prompt: str, execution_time: float, code: str) -> None:
    """
    Log details about a successful execution.
    
    Args:
        result: The output of the execution
        prompt: The prompt that generated the code
        execution_time: The execution time in seconds
        code: The executed code
    """
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "execution_time_seconds": execution_time,
        "answer": result.get("answer", ""),
        "chart_type": result.get("chart_type", "none"),
        "confidence": result.get("confidence", 0),
        "code": code
    }
    
    # Log the execution details
    logger.info(f"Agent execution completed: {json.dumps(log_entry, indent=2)}")

def code_execution_in_sandbox(
    code: str,
    df: pd.DataFrame,
    schema: Dict[str, Any] = None,
    llm_service_func: Optional[Callable] = None,
    original_prompt: Optional[str] = None,
    enable_retry: bool = True,
    config: Optional[SandboxConfig] = None
) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment.
    
    Args:
        code: Python code string to execute
        df: Pandas DataFrame to provide to the code
        schema: JSON schema to validate the output against
        llm_service_func: Function to call LLM service for retries (required if enable_retry=True)
        original_prompt: Original prompt that generated the code (required if enable_retry=True)
        enable_retry: Whether to enable automatic retry with modified prompt
        config: Configuration for sandbox execution
        
    Returns:
        Dictionary containing the validated output
        
    Raises:
        SandboxExecutionError: If execution fails
        SchemaValidationError: If output validation fails
        ValueError: If retry is enabled but required parameters are missing
    """
    # Validate parameters for retry
    if enable_retry and (llm_service_func is None or original_prompt is None):
        raise ValueError("llm_service_func and original_prompt are required when enable_retry=True")
    
    # Use default schema if not provided
    schema = schema or DEFAULT_OUTPUT_SCHEMA
    config = config or SandboxConfig()
    
    start_time = time.time()
    
    try:
        # Execute the code
        result = execute_code_in_sandbox(code, df, schema, config)
        execution_time = time.time() - start_time
        
        # Log the successful execution
        if original_prompt:
            log_execution(result, original_prompt, execution_time, code)
        
        return result
        
    except (SandboxExecutionError, SchemaValidationError) as e:
        # Attempt retry if enabled
        if enable_retry and llm_service_func and original_prompt:
            return retry_with_modified_prompt(
                execute_code_in_sandbox,
                e,
                df,
                schema,
                llm_service_func,
                original_prompt
            )
        else:
            # Re-raise the error if retry is not enabled
            raise