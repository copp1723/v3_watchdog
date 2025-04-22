"""
Sandboxed Execution System for Watchdog AI insights.
Provides a secure environment for running insight generation code.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import ast
import logging
import resource
import threading
from contextlib import contextmanager
import traceback
from datetime import datetime

from .contracts import InsightContract, InsightContractEnforcer

logger = logging.getLogger(__name__)

class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code for potential security issues."""
    
    def __init__(self):
        self.issues = []
        self.imported_modules = set()
        
    def visit_Import(self, node):
        """Check imported modules."""
        for name in node.names:
            self.imported_modules.add(name.name)
            if name.name not in ALLOWED_IMPORTS:
                self.issues.append(f"Unauthorized import: {name.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check imported modules."""
        if node.module not in ALLOWED_IMPORTS:
            self.issues.append(f"Unauthorized import: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_FUNCTIONS:
                self.issues.append(f"Unauthorized function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_METHODS:
                self.issues.append(f"Unauthorized method call: {node.func.attr}")
        self.generic_visit(node)

# Security Configuration
ALLOWED_IMPORTS = {
    'pandas', 'numpy', 'datetime', 'typing',
    'collections', 'itertools', 'functools'
}

BLOCKED_FUNCTIONS = {
    'eval', 'exec', 'compile', 'open', 'input',
    'subprocess', 'os.system', 'os.popen'
}

BLOCKED_METHODS = {
    'system', 'popen', 'shell', 'eval', 'exec',
    'read', 'write', 'delete', 'remove'
}

# Resource Limits (in bytes)
MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB
TIME_LIMIT = 30  # 30 seconds

@contextmanager
def resource_limits():
    """Set resource limits for the execution."""
    # Set memory limit
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    
    # Set CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, (TIME_LIMIT, TIME_LIMIT))
    
    try:
        yield
    finally:
        # Reset limits to default
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

class TimeoutError(Exception):
    """Raised when execution exceeds time limit."""
    pass

class SandboxedExecution:
    """Provides a secure environment for running insight generation code."""
    
    def __init__(self, contract_enforcer: InsightContractEnforcer):
        """Initialize the sandbox."""
        self.contract_enforcer = contract_enforcer
        self.execution_context = {}
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for security issues.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            return {
                "is_safe": len(analyzer.issues) == 0,
                "issues": analyzer.issues,
                "imported_modules": analyzer.imported_modules
            }
        except Exception as e:
            return {
                "is_safe": False,
                "issues": [f"Code analysis error: {str(e)}"],
                "imported_modules": set()
            }
    
    def _run_with_timeout(self, func, args=None, kwargs=None, timeout=TIME_LIMIT):
        """Run a function with a timeout."""
        result = {"completed": False, "error": None, "value": None}
        
        def worker():
            try:
                result["value"] = func(*(args or []), **(kwargs or {}))
                result["completed"] = True
            except Exception as e:
                result["error"] = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Execution exceeded {timeout} seconds")
        
        if result["error"]:
            raise result["error"]
        
        return result["value"]
    
    def run_insight(self, code: str, data: pd.DataFrame,
                   contract: InsightContract) -> Dict[str, Any]:
        """
        Run insight generation code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            data: Input DataFrame
            contract: Insight contract to enforce
            
        Returns:
            Generated insight output
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            input_validation = self.contract_enforcer.validate_input(data, contract)
            if not input_validation["is_valid"]:
                return {
                    "error": "Input validation failed",
                    "details": input_validation,
                    "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Analyze code
            analysis = self.analyze_code(code)
            if not analysis["is_safe"]:
                return {
                    "error": "Code security check failed",
                    "details": analysis,
                    "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Prepare execution environment
            local_vars = {
                'pd': pd,
                'np': np,
                'data': data.copy(),  # Use copy to prevent modifications
                'datetime': datetime
            }
            
            # Run code in sandbox
            with resource_limits():
                try:
                    # Compile code to prevent use of exec/eval
                    compiled_code = compile(code, '<string>', 'exec')
                    
                    # Run with timeout
                    self._run_with_timeout(
                        exec,
                        args=(compiled_code, local_vars),
                        timeout=TIME_LIMIT
                    )
                    
                    # Get result from local vars
                    if 'result' not in local_vars:
                        raise ValueError("Code did not produce a 'result' variable")
                    
                    output = local_vars['result']
                    
                except Exception as e:
                    return {
                        "error": "Execution error",
                        "details": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        },
                        "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                    }
            
            # Validate output
            output_validation = self.contract_enforcer.validate_output(output, contract)
            if not output_validation["is_valid"]:
                return {
                    "error": "Output validation failed",
                    "details": output_validation,
                    "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Add execution metadata
            output["execution_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            output["execution_context"] = {
                "memory_used": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "imported_modules": list(analysis["imported_modules"])
            }
            
            return output
            
        except Exception as e:
            return {
                "error": "Unexpected error",
                "details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                },
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }