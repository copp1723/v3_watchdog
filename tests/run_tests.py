#!/usr/bin/env python
import unittest
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import test modules
from tests.nova_act_test import TestNovaScheduler, TestNovaFallback, TestWatchdogUpload

def run_tests():
    """Run all test cases for Nova Act integration"""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestNovaScheduler))
    test_suite.addTest(unittest.makeSuite(TestNovaFallback))
    test_suite.addTest(unittest.makeSuite(TestWatchdogUpload))
    
    # Run the test suite
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == "__main__":
    # Run the tests and get the result
    result = run_tests()
    
    # Exit with the number of failures and errors
    sys.exit(result) 