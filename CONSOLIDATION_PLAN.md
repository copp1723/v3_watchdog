# Code Structure Consolidation Plan

## Current Issues

The codebase currently has a duplicated structure with code spread across both `src/` and `src/watchdog_ai/` directories. This creates several problems:

1. **Unclear import paths**: Developers might be confused about which module to import from
2. **Duplicate functionality**: Same or similar functionality may exist in both locations
3. **Maintenance overhead**: Changes need to be synchronized across multiple files
4. **Package structure confusion**: Unclear packaging makes deployment more complicated

## Consolidation Approach

We will adopt a proper Python package structure with `src/watchdog_ai/` as the main package, following these principles:

1. **Single Source of Truth**: All code should exist in only one location
2. **Clear Module Hierarchy**: Organized modules with clear responsibilities
3. **Standard Package Structure**: Follow Python packaging best practices
4. **Backward Compatibility**: Provide import compatibility layers where needed

## Target Structure

```
src/
└── watchdog_ai/
    ├── __init__.py
    ├── analytics/           # Data analysis modules
    ├── api/                 # API endpoints
    ├── auth/                # Authentication and user management
    ├── core/                # Core application functionality
    ├── data/                # Data processing and management
    ├── insights/            # Insight generation and management
    ├── llm/                 # Language model integration
    ├── models/              # Data models and schemas
    ├── scheduler/           # Scheduling functionality
    ├── tests/               # Package-level tests
    ├── ui/                  # User interface components
    │   ├── components/
    │   ├── pages/
    │   └── layouts/
    ├── utils/               # Utility functions
    └── validators/          # Data validation
```

## Implementation Plan

### Phase 1: Preparation

1. Create a detailed inventory of all modules in both `src/` and `src/watchdog_ai/`
2. Identify duplicates and determine which version to keep
3. Map dependencies between modules
4. Design a migration strategy that minimizes disruption

### Phase 2: Consolidation

1. Migrate code module by module, prioritizing core functionality
2. Update import statements in all files
3. Create compatibility layers for backward compatibility
4. Run tests after each module migration

### Phase 3: Cleanup

1. Remove redundant code
2. Update documentation
3. Update build and packaging scripts
4. Finalize the new structure

## Task List for Implementation

### Phase 1: Preparation Tasks

- [ ] 1.1 Create a complete inventory of modules in `src/` 
- [ ] 1.2 Create a complete inventory of modules in `src/watchdog_ai/`
- [ ] 1.3 Identify duplicate functionality and decide which to keep
- [ ] 1.4 Map dependencies between modules using tools like `import-graph`
- [ ] 1.5 Create a detailed migration sequence based on dependencies

### Phase 2: Consolidation Tasks

- [ ] 2.1 Set up proper package structure in `src/watchdog_ai/`
- [ ] 2.2 Consolidate core modules first:
  - [ ] 2.2.1 Move/merge core functionality
  - [ ] 2.2.2 Update import statements
  - [ ] 2.2.3 Run tests to verify functionality
- [ ] 2.3 Consolidate analytics modules:
  - [ ] 2.3.1 Move/merge analytics functionality
  - [ ] 2.3.2 Update import statements
  - [ ] 2.3.3 Run tests to verify functionality
- [ ] 2.4 Consolidate insight modules:
  - [ ] 2.4.1 Move/merge insight functionality
  - [ ] 2.4.2 Update import statements
  - [ ] 2.4.3 Run tests to verify functionality
- [ ] 2.5 Consolidate LLM modules:
  - [ ] 2.5.1 Move/merge LLM functionality
  - [ ] 2.5.2 Update import statements
  - [ ] 2.5.3 Run tests to verify functionality
- [ ] 2.6 Consolidate validator modules:
  - [ ] 2.6.1 Move/merge validator functionality
  - [ ] 2.6.2 Update import statements
  - [ ] 2.6.3 Run tests to verify functionality
- [ ] 2.7 Consolidate UI components:
  - [ ] 2.7.1 Move/merge UI components
  - [ ] 2.7.2 Update import statements
  - [ ] 2.7.3 Run tests to verify functionality
- [ ] 2.8 Consolidate utility functions:
  - [ ] 2.8.1 Move/merge utility functions
  - [ ] 2.8.2 Update import statements
  - [ ] 2.8.3 Run tests to verify functionality
- [ ] 2.9 Consolidate remaining modules

### Phase 3: Cleanup Tasks

- [ ] 3.1 Remove redundant files from `src/` root
- [ ] 3.2 Update package __init__.py files with proper exports
- [ ] 3.3 Update documentation with new import paths
- [ ] 3.4 Update setup.py and packaging configuration
- [ ] 3.5 Create or update package README files
- [ ] 3.6 Update CI/CD pipeline configurations
- [ ] 3.7 Run comprehensive test suite
- [ ] 3.8 Create compatibility layer for backward compatibility (if needed)

## Backward Compatibility Strategy

To maintain backward compatibility during the transition:

1. Create compatibility modules that re-export from the new locations
2. Deprecate old import paths with warnings
3. Document new import paths in the codebase
4. Set a timeline for removing compatibility layers

## Testing Strategy

1. Run tests after each module migration
2. Create additional tests for edge cases
3. Test imports from both old and new paths during transition
4. Verify functionality across the entire application

## Documentation Updates

1. Update README.md with new package structure
2. Update import examples in documentation
3. Create module-level documentation
4. Update API documentation

## Timeline

- Phase 1: 1-2 days
- Phase 2: 3-5 days (depending on complexity)
- Phase 3: 1-2 days

Total estimated time: 5-9 days

