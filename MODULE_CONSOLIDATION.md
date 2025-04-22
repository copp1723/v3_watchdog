# Module Consolidation Inventory

This document identifies modules that need to be consolidated between `src/` and `src/watchdog_ai/` directories, determining the primary module to keep and the duplicate to merge or remove.

## Consolidation Principles

1. **Favor the `src/watchdog_ai/` structure** as our target package
2. **Merge functionality** from duplicate modules rather than simply discarding one
3. **Consider newer code** as potentially more up-to-date
4. **Keep better structure** when deciding between similar implementations

## Core Modules

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Application Entry | `src/watchdog_ai/app.py` | `src/core/app.py` | Merge functionality, keeping watchdog_ai structure |
| Server | `src/watchdog_ai/core/` | `src/core/server.py` | Move into appropriate core submodule |
| Types | `src/watchdog_ai/models/` | `src/core/watchdog_types.py` | Merge types into appropriate model definitions |
| Celery | `src/watchdog_ai/workers/` | `src/core/celery_app.py` | Move worker functionality to workers/ directory |

## Insights Modules

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Insight Engine | `src/watchdog_ai/insights/engine.py` | `src/insights/` various files | Consolidate core insight functionality |
| Intent Management | `src/watchdog_ai/insights/intent_manager.py` | `src/insights/intent_manager.py` | Compare implementations and merge |
| Insight Conversation | `src/watchdog_ai/insights/insight_conversation.py` | `src/insights/insight_conversation.py` | Merge conversation handling logic |
| Digest System | `src/watchdog_ai/insights/` | `src/insights/digest_system.py` | Add digest functionality to insights package |
| Insight Card | `src/watchdog_ai/insights/` | `src/insights/insight_card.py` | Merge with core/insights/card.py if appropriate |
| Insight Flow | `src/watchdog_ai/insights/` | `src/insights/insight_flow.py` | Consolidate flow logic |
| Feedback Engine | `src/watchdog_ai/insights/feedback_engine.py` | `src/insights/feedback_engine.py` | Compare and merge |
| Traceability | `src/watchdog_ai/insights/traceability.py` | `src/insights/traceability.py` | Compare and select best implementation |

## UI Components

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Debug Panel | `src/watchdog_ai/ui/components/debug_panel.py` | `src/ui/components/query_debug_panel.py` | Merge functionality |
| Fallback Renderer | `src/watchdog_ai/insights/fallback_renderer.py` | `src/ui/components/fallback_renderer.py` | Determine best location based on functionality |
| Notification Demo | `src/watchdog_ai/ui/` | `src/ui/components/notification_demo.py` | Add to appropriate UI component |
| UI Pages | `src/watchdog_ai/ui/pages/` | `src/ui/pages/` | Consolidate page implementations |

## LLM Engine

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| LLM Engine | `src/watchdog_ai/llm/llm_engine.py` | `src/llm/llm_engine.py` | Compare and merge functionality |
| Query Processing | `src/watchdog_ai/llm/query_processor.py` | `src/llm/query_rewriter.py` | Consolidate query handling |
| Direct Processing | `src/watchdog_ai/insights/direct_llm_query.py` | `src/llm/direct_processors.py` | Merge direct query handling |

## Validators

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Data Validation | `src/watchdog_ai/core/data_validator.py` | `src/validators/data_validation.py` | Consolidate validation logic |
| Schema Management | `src/watchdog_ai/schema/` | `src/validators/schema_manager.py` | Merge schema management |
| Rule Engine | `src/watchdog_ai/` | `src/validators/rule_engine.py` | Add to appropriate location in package |
| Validator Service | `src/watchdog_ai/` | `src/validators/validator_service.py` | Create validators package in watchdog_ai |
| Column Mapping | `src/watchdog_ai/utils/llm_column_mapper.py` | `src/validators/column_mapper.py` | Compare and merge |

## Utils

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Chart Utils | `src/watchdog_ai/core/visualization/chart_utils.py` | `src/utils/chart_utils.py` | Compare and consolidate |
| Data I/O | `src/watchdog_ai/utils/data_parser.py` | `src/utils/data_io.py` | Merge data handling functionality |
| Logging | `src/watchdog_ai/core/logging_config.py` | `src/utils/logging_config.py` | Consolidate logging configuration |
| Error Handling | `src/watchdog_ai/utils/errors.py` | `src/utils/errors.py` | Compare and merge error handling |
| PDF Generation | `src/watchdog_ai/utils/pdf_extractor.py` | `src/utils/pdf_generator.py` | Combine PDF utilities |
| Data Normalization | `src/watchdog_ai/utils/data_normalization.py` | `src/utils/data_normalization.py` | Compare and merge |
| Schema Handling | `src/watchdog_ai/utils/adaptive_schema.py` | `src/utils/adaptive_schema.py` | Compare implementations |

## Analytics

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Semantic Layer | `src/watchdog_ai/analysis/` | `src/analytics/semantic_layer.py` | Add to analysis package or create analytics |
| Trend Analysis | `src/watchdog_ai/` | `src/analytics/trend_analysis.py` | Create analytics package if not exists |
| Precision Scoring | `src/watchdog_ai/insights/precision_scoring.py` | `src/analytics/precision_scoring.py` | Determine best location |

## Nova Act Integration

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Nova Act Integration | `src/watchdog_ai/integrations/crm/nova_act.py` | `src/nova_act/` | Compare functionality and merge as needed |
| Scheduler | `src/watchdog_ai/workers/scheduler.py` | `src/nova_act/scheduler.py` | Consolidate scheduler functionality |

## Authentication/Sessions

| Functionality | Primary Module (Keep) | Duplicate Module (Merge) | Notes |
|---------------|----------------------|------------------------|-------|
| Session Management | `src/watchdog_ai/` | `src/auth/session.py` | Create auth package in watchdog_ai |
| User Management | `src/watchdog_ai/` | `src/auth/user_management.py` | Create auth package in watchdog_ai |

## Consolidation Sequence

Based on dependencies, here's the recommended sequence for module consolidation:

1. **Core and Models** - These provide the foundation
   - Types, Models, Configuration

2. **Utils** - These are used by many other modules
   - Logging, Errors, Data I/O, Schema Utilities

3. **Authentication** - User and session management

4. **LLM Engine** - Core language model functionality

5. **Validators** - Data validation logic

6. **Analytics** - Analysis capabilities

7. **Insights** - Insight generation and processing

8. **UI Components** - User interface elements

9. **Integration** - External integrations like Nova Act

10. **Workers** - Background processes and scheduling

## Next Steps

1. Begin detailed code comparison of each module pair
2. Document specific functions, classes, and methods to merge
3. Create integration tests to verify functionality before and after merging
4. Start consolidation with least dependent modules first

