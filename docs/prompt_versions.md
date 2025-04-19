# Prompt Version History

This document tracks the versioning and changes to prompt templates used in the Watchdog AI system.

## Column Mapping Prompt

| Version | Date | Description |
|---------|------|-------------|
| 1.3.0 | 2025-04-19 | Added version comment, enhanced documentation for Jeopardy-style reasoning, improved clarification protocol, and tuned confidence thresholds. |
| 1.0.0 | 2025-04-01 | Initial version of column_mapping.tpl with basic Jeopardy-style mapping functionality. |

### Version 1.3.0 Changes
- Added version comment at the top of the prompt
- Enhanced documentation for developers
- Improved clarification protocol with better question formatting
- Added explicit handling of lead-source values as column headers
- Added support for confidence scoring (0.0-1.0)
- Set 0.7 as the threshold for automatic mapping without clarification
- Added additional examples for different column naming patterns

## Intent Detection Prompt

| Version | Date | Description |
|---------|------|-------------|
| 1.1.0 | 2025-04-19 | Added support for lead source queries, enhanced function calling format, and additional examples. |
| 1.0.0 | 2025-03-15 | Initial version with basic intent classification. |

### Version 1.1.0 Changes
- Added specific handling for lead source queries
- Enhanced validation rules for intent matching
- Added function calling wrapper for improved structure
- Added 10 new example queries with their intent classifications

## Guidelines for Prompt Versioning

When updating prompt templates, please follow these guidelines:

1. **Version Format**: Use semantic versioning (MAJOR.MINOR.PATCH)
   - MAJOR: Breaking changes that completely alter the output structure
   - MINOR: Non-breaking enhancements, new capabilities or examples
   - PATCH: Bug fixes, clarifications, or rephrasing without functional changes

2. **Update Process**:
   - Update the version comment at the top of the prompt file
   - Document the changes in this file with a new entry
   - Include specific details about what was changed and why
   - Run tests to ensure the changes maintain or improve performance

3. **Documenting Changes**:
   - List all significant changes in bullet points
   - Include before/after examples if helpful
   - Note any new behaviors or capabilities
   - Document any changes in confidence thresholds or scoring mechanisms