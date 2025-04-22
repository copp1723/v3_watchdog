# Watchdog AI Cleanup Report

Generated: 2025-04-21 20:09:58

## Summary

### Code Issues

- Total issues: **29**
- Production code issues: **28**
- Non-production code issues: **1**
- TODO/FIXME count: **24**
- Deprecated code count: **5**

### Documentation Issues

- Total issues: **0**
- Broken links: **0**

### Legacy Artifacts

- Total artifacts: **32**
- Unused artifacts: **31**
- Large unused artifacts (>2MB): **0**
- Total unused size: **0.28 MB**

### CI YAML Issues

- Files with commented sections: **0**
- Total commented lines: **0**

### Quality Checks

- Linting: **FAIL**
- Unit tests: **FAIL**
- Code coverage: **FAIL** (4.0%)
- Visual tests: **FAIL**

## Detailed Code Issues

| File | Line | Type | Content | Production |
|------|------|------|---------|------------|
| src/watchdog_ai/llm_engine.py | 18 | TODO/FIXME | self.chat = MockChat() if use_mock else None  #... | Yes |
| src/watchdog_ai/llm_engine.py | 26 | TODO/FIXME | # TODO: Add real LLM implementation | Yes |
| src/utils/cleanup.py | 6 | TODO/FIXME | 1. Scanning for TODO/FIXME/deprecated markers | Yes |
| src/utils/cleanup.py | 6 | TODO/FIXME | 1. Scanning for TODO/FIXME/deprecated markers | Yes |
| src/utils/cleanup.py | 93 | TODO/FIXME | "TODO", | Yes |
| src/utils/cleanup.py | 94 | TODO/FIXME | "FIXME", | Yes |
| src/utils/cleanup.py | 95 | Deprecated | "@deprecated", | Yes |
| src/utils/cleanup.py | 96 | Deprecated | "# Deprecated", | Yes |
| src/utils/cleanup.py | 97 | Deprecated | "// Deprecated", | Yes |
| src/utils/cleanup.py | 98 | Deprecated | "/* Deprecated", | Yes |
| src/utils/cleanup.py | 99 | Deprecated | "This is deprecated", | Yes |
| src/utils/cleanup.py | 409 | TODO/FIXME | "todo_fixme_count": sum(1 for i in code_issues ... | Yes |
| src/utils/cleanup.py | 409 | TODO/FIXME | "todo_fixme_count": sum(1 for i in code_issues ... | Yes |
| src/utils/cleanup.py | 502 | TODO/FIXME | f.write(f"- TODO/FIXME count: **{code_summary['... | Yes |
| src/utils/cleanup.py | 502 | TODO/FIXME | f.write(f"- TODO/FIXME count: **{code_summary['... | Yes |
| src/utils/cleanup.py | 572 | TODO/FIXME | f.write("- **High Priority**: Clean up the TODO... | Yes |
| src/utils/cleanup.py | 572 | TODO/FIXME | f.write("- **High Priority**: Clean up the TODO... | Yes |
| src/utils/cleanup.py | 610 | TODO/FIXME | Scan the codebase for TODO, FIXME, and deprecat... | Yes |
| src/utils/cleanup.py | 610 | TODO/FIXME | Scan the codebase for TODO, FIXME, and deprecat... | Yes |
| src/utils/cleanup.py | 635 | TODO/FIXME | issue_type = 'TODO/FIXME' if any(k in keyword f... | Yes |
| src/utils/cleanup.py | 635 | TODO/FIXME | issue_type = 'TODO/FIXME' if any(k in keyword f... | Yes |
| src/validators/validator_service.py | 187 | TODO/FIXME | # TODO: Get LLM engine instance properly (poten... | Yes |
| src/validators/validator_service.py | 205 | TODO/FIXME | # TODO: Implement UI interaction for clarificat... | Yes |
| src/validators/validator_service.py | 231 | TODO/FIXME | # TODO: Make confidence threshold configurable? | Yes |
| src/validators/validator_service.py | 507 | TODO/FIXME | # TODO: Potentially add stricter validation if ... | Yes |
| src/validators/validator_service.py | 573 | TODO/FIXME | # TODO: Add option for manual mapping override? | Yes |
| src/validators/validation_profile.py | 159 | TODO/FIXME | # TODO: Add unit tests for these rules | Yes |
| src/validators/validation_profile.py | 384 | TODO/FIXME | # TODO: Add more rules based on docs/validation... | Yes |
| tests/test_prompt_bank.py | 114 | TODO/FIXME | # TODO: Manual Evaluation | No |

## Unused Artifacts

| Path | Size |
|------|------|
| archive/old-tests/__pycache__/test_insight_conversation_enhanced.cpython-313-pytest-8.3.5.pyc | 29.46 KB |
| archive/old-tests/__pycache__/test_end_to_end_enhanced.cpython-313-pytest-8.3.5.pyc | 26.29 KB |
| archive/old-tests/__pycache__/test_file_upload_enhanced.cpython-313-pytest-8.3.5.pyc | 21.08 KB |
| archive/old-versions/insight_card_improved.py | 18.68 KB |
| legacy/insight_card_consolidated.py | 18.68 KB |
| legacy/consolidated/insight_card_consolidated.py | 18.68 KB |
| legacy/insight_conversation_consolidated.py | 17.99 KB |
| legacy/consolidated/insight_conversation_consolidated.py | 17.99 KB |
| archive/old-versions/insight_flow_enhanced.py | 14.40 KB |
| archive/old-versions/insight_conversation_enhanced.py | 13.92 KB |
| archive/old-tests/test_end_to_end_enhanced.py | 11.50 KB |
| archive/old-tests/test_insight_conversation_enhanced.py | 9.02 KB |
| archive/old-tests/test_app_enhanced.py | 7.62 KB |
| archive/old-versions/insight_conversation_new.py | 7.32 KB |
| legacy/duplicate_files/__pycache__/test_nova_act_retries_2.cpython-313-pytest-8.3.5.pyc | 6.32 KB |
| archive/old-tests/test_file_upload_enhanced.py | 6.17 KB |
| archive/.DS_Store | 6.00 KB |
| legacy/.DS_Store | 6.00 KB |
| legacy/insights/traceability.py | 4.27 KB |
| archive/old-versions/chat_interface_enhanced.py | 4.21 KB |
| archive/old-versions/data_upload_enhanced.py | 3.76 KB |
| legacy/duplicate_files/__pycache__/test_session_ttl_2.cpython-313-pytest-8.3.5.pyc | 3.30 KB |
| legacy/insights/intent_processor.py | 3.01 KB |
| legacy/duplicate_files/utils_audit_log_2.py | 2.60 KB |
| legacy/duplicate_files/test_nova_act_retries_2.py | 1.84 KB |
| legacy/duplicate_files/test_session_ttl_2.py | 1.49 KB |
| legacy/duplicate_files/baselines 2.py | 1.44 KB |
| legacy/duplicate_files/session 2.py | 1.39 KB |
| legacy/duplicate_files/audit_log 2.py | 1.17 KB |
| legacy/duplicate_files/utils_logging_2.py | 802.00 B |
| legacy/duplicate_files/base_validator_2.py | 372.00 B |

## Recommendations

- **High Priority**: Clean up the TODO/FIXME and deprecated code in production files
- **High Priority**: Fix linting issues and failing tests
- **Medium Priority**: Improve test coverage to meet the 85.0% threshold
