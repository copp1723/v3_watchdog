import unittest
from unittest.mock import patch
import streamlit as st

from src.utils.session import get_session_id


class TestSessionPersistence(unittest.TestCase):
    def test_session_id_persistence(self):
        fake_session = {}
        # First run: initialize session state
        with patch.dict(st.session_state, fake_session, clear=True):
            session_id_1 = get_session_id()
        # Second run: using the same fake session dictionary to simulate persistence
        with patch.dict(st.session_state, fake_session, clear=True):
            session_id_2 = get_session_id()
        self.assertEqual(session_id_1, session_id_2, "Session IDs should persist across runs")


if __name__ == '__main__':
    unittest.main() 