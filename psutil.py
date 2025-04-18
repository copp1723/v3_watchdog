"""
Stub psutil module for compatibility with performance tests.
This provides minimal Process class with memory_info and num_handles methods.
"""
import os

class Process:
    def __init__(self, pid=None):
        # pid argument ignored in stub
        pass

    def memory_info(self):
        class mem:
            rss = 0  # Return 0 bytes used
        return mem()

    def num_handles(self):
        # Return 0 handles in stub
        return 0

# Expose Process at module level 