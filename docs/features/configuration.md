# Configuration Management

This document covers the configuration system for Watchdog AI, including environment variables, thresholds, feature flags, and security features.

## Getting Started

Watchdog AI uses a flexible configuration system that supports multiple environments (development, staging, production) and secure handling of sensitive information like API keys. Configuration is primarily managed through environment variables, with additional support for `.env` files in different environments.

The configuration system is implemented in `src/utils/config.py` and provides methods to access configuration values safely, perform validation, and handle encrypted secrets.

## Environment Variables

### Core Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM functionality | None | Yes (if `USE_MOCK=false`) |
| `AGENTOPS_API_KEY` | AgentOps API key for monitoring | None | No |
| `USE_MOCK` | Use mock responses instead of real LLM calls | `false` | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` | No |
| `MAX_UPLOAD_SIZE_MB` | Maximum upload size in megabytes | `100` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `TESTING` | Enable testing mode | `false` | No |

### Redis Cache Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_CACHE_ENABLED` | Enable Redis caching | `true` | No |
| `REDIS_HOST` | Redis server hostname | `localhost` | If Redis enabled |
| `REDIS_PORT` | Redis server port | `6379` | If Redis enabled |
| `REDIS_DB` | Redis database number | `0` | No |
| `COLUMN_MAPPING_CACHE_TTL` | Column mapping cache time-to-live in seconds | `86400` (1 day) | No |
| `COLUMN_MAPPING_CACHE_PREFIX` | Prefix for Redis keys | `watchdog:column_mapping:` | No |

### Feature Flags

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DROP_UNMAPPED_COLUMNS` | Automatically drop unmapped columns after clarification | `false` | No |

## Default Thresholds

The following thresholds are defined in `src/utils/config.py`:

| Threshold | Value | Description |
|-----------|-------|-------------|
| `MIN_CONFIDENCE_TO_AUTOMAP` | `0.7` | Minimum confidence threshold for automatic column mapping |

Rate limiting defaults:
- Default rate limit window: 5 minutes
- Default maximum requests: 100

## Environment-Specific Configuration

Watchdog AI supports different environments through environment-specific `.env` files:

### Development Environment

Use `.env.development` for local development settings. Copy `.env.template` to create this file:

```bash
cp config/env/.env.template config/env/.env.development
```

Edit the file to add your API keys and configure development-specific settings:

```
# Example development settings
OPENAI_API_KEY=your-openai-key
USE_MOCK=true  # Use mock responses for development
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Environment

For production, use `.env.production` with stricter settings:

```
# Production settings example
USE_MOCK=false  # Use real LLM in production
OPENAI_API_KEY=your-openai-key
DEBUG=false
LOG_LEVEL=INFO
REDIS_CACHE_ENABLED=true
MAX_UPLOAD_SIZE_MB=50  # More restricted upload limit
```

### Loading Environment Configuration

The system prioritizes environment variables in the following order:
1. Operating system environment variables
2. Environment-specific `.env` file
3. Default values in the code

## Configuration Validation and Security

### Validation

The configuration system validates critical settings to ensure the application can function correctly. Validation includes:

- Checking required API keys based on configuration
- Validating API key formats
- Ensuring valid values for boolean and integer settings
- Type checking for configuration values

When validation fails, the system raises a `ConfigurationError` with details about the invalid configuration.

### Secure Secrets Management

Sensitive information like API keys can be stored securely using the encrypted secrets feature:

```python
# Store a sensitive value securely
config.set_secret("API_KEY", "your-api-key")

# Retrieve the decrypted value
api_key = config.get_secret("API_KEY")
```

The encryption system uses:
- Fernet symmetric encryption (AES-128 in CBC mode with PKCS7 padding)
- Key stored in a `.secret_key` file with secure permissions (0600)
- Automatic key generation if none exists

### Request Signing and Verification

For secure API communication, the configuration system provides methods to sign and verify requests:

```python
# Generate a signature for request data
signature = config.generate_request_signature(request_data)

# Verify the signature of incoming data
is_valid = config.verify_request_signature(request_data, signature)
```

## Working with Configuration

### Using Configuration in Code

The global `config` instance is available for accessing configuration values:

```python
from src.utils.config import config

# Get a configuration value with a default fallback
debug_mode = config.get_bool("DEBUG", False)

# Get an integer value
max_size = config.get_int("MAX_UPLOAD_SIZE_MB", 50)

# Get a sensitive configuration value
api_key = config.get_secret("OPENAI_API_KEY")
```

### Rate Limiting

The configuration system includes built-in rate limiting functionality:

```python
# Check if a request is allowed under rate limits
if config.check_rate_limit("user_123"):
    # Process the request
else:
    # Return rate limit exceeded error
```

## Best Practices

1. **Never commit sensitive values**: Keep all API keys and secrets out of version control
2. **Use environment-specific configurations**: Maintain separate `.env` files for different environments
3. **Validate all inputs**: Always validate configuration values before using them
4. **Set reasonable defaults**: Provide sensible defaults for non-critical configuration
5. **Document all options**: Keep this documentation updated when adding new configuration options

