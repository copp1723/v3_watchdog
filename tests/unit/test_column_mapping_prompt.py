import unittest
import os

class TestColumnMappingPrompt(unittest.TestCase):

    def setUp(self):
        """Set up test variables."""
        self.prompt_file_path = "src/insights/prompts/column_mapping.tpl"
        self.required_sections = [
            "# Watchdog AI Data‑Upload Column Mapping Prompt",
            "## 1. Canonical Schema",
            "## 2. Jeopardy‑Style Mapping Process",
            "## 3. Clarification Protocol",
            "## 4. Response Schema",
            "## 5. Few‑Shot Examples",
            "Dataset columns: {{columns}}"
        ]

    def test_prompt_file_exists(self):
        """Test that the column_mapping.tpl file exists."""
        self.assertTrue(os.path.exists(self.prompt_file_path),
                        f"Prompt file not found at: {self.prompt_file_path}")

    def test_prompt_file_content(self):
        """Test that the prompt file contains all required sections."""
        self.assertTrue(os.path.exists(self.prompt_file_path), "Prompt file missing, cannot test content.")
        
        with open(self.prompt_file_path, 'r') as f:
            content = f.read()
            
        missing_sections = []
        for section in self.required_sections:
            if section not in content:
                missing_sections.append(section)
                
        self.assertFalse(missing_sections, 
                         f"Prompt file {self.prompt_file_path} is missing required sections:\n" + 
                         "\n".join(missing_sections))

    def test_prompt_version_comment(self):
        """Test that the prompt file starts with the version comment."""
        self.assertTrue(os.path.exists(self.prompt_file_path), "Prompt file missing, cannot test content.")
        
        with open(self.prompt_file_path, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            
        self.assertTrue(first_line.startswith("# column_mapping.tpl — version"),
                        f"First line does not start with version comment: '{first_line}'")
        self.assertEqual(second_line, "# Unified Jeopardy‑style mapping prompt...",
                         f"Second line does not match expected comment: '{second_line}'")

if __name__ == '__main__':
    unittest.main() 