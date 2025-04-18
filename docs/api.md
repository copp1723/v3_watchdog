# V3 Watchdog AI API Reference

This document outlines the available API endpoints for the V3 Watchdog AI platform. The API allows programmatic access to insights, reports, and data management features.

## Authentication

All API requests require authentication using JWT tokens.

```
Authorization: Bearer <your_token>
```

To obtain a token, use the `/auth/token` endpoint with your credentials.

## Base URL

The base URL for all API endpoints is:

```
https://api.watchdogai.com/v1
```

For development environments:

```
http://localhost:8000/v1
```

## Endpoints

### Authentication

#### Generate Token

```
POST /auth/token
```

**Parameters:**

| Name     | Type   | Description            |
|----------|--------|------------------------|
| username | string | Your username          |
| password | string | Your account password |

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Insights

#### Get Insights

```
GET /insights
```

**Query Parameters:**

| Name     | Type   | Description                      |
|----------|--------|----------------------------------|
| type     | string | Filter by insight type (optional) |
| limit    | number | Number of insights to return     |
| offset   | number | Number of insights to skip       |

**Response:**

```json
{
  "insights": [
    {
      "id": "6d9e73f5-3d41-4b88-99d6-6946c1843f39",
      "type": "sales_performance",
      "title": "Sales Performance Analysis",
      "summary": "Sales increased by 12% compared to previous period",
      "created_at": "2023-04-01T14:25:30Z"
    },
    ...
  ],
  "total": 120,
  "limit": 10,
  "offset": 0
}
```

#### Get Insight by ID

```
GET /insights/{insight_id}
```

**Response:**

```json
{
  "id": "6d9e73f5-3d41-4b88-99d6-6946c1843f39",
  "type": "sales_performance",
  "title": "Sales Performance Analysis",
  "summary": "Sales increased by 12% compared to previous period",
  "created_at": "2023-04-01T14:25:30Z",
  "data": {
    "metrics": {
      "total_sales": 523,
      "average_gross": 3245.50,
      "top_performer": "John Doe"
    },
    "charts": [
      {
        "title": "Sales by Department",
        "type": "pie",
        "data": {...}
      }
    ],
    "recommendations": [
      "Focus on increasing back-end product penetration",
      "Review underperforming lead sources"
    ]
  }
}
```

#### Generate Custom Insight

```
POST /insights
```

**Request Body:**

```json
{
  "type": "custom",
  "prompt": "Show me the conversion rate by lead source for the last month",
  "parameters": {
    "timeframe": "last_month",
    "metrics": ["conversion_rate", "total_leads", "total_sales"]
  }
}
```

**Response:**

```json
{
  "id": "8f7e65d4-2c31-5a79-88e5-5835d1932f48",
  "type": "custom",
  "title": "Lead Source Conversion Analysis",
  "summary": "Website leads show highest conversion at 12.3%",
  "created_at": "2023-04-15T09:12:45Z",
  "data": {...}
}
```

### Reports

#### List Reports

```
GET /reports
```

**Query Parameters:**

| Name       | Type   | Description                      |
|------------|--------|----------------------------------|
| type       | string | Filter by report type            |
| status     | string | Filter by status                 |
| created_by | string | Filter by creator                |
| limit      | number | Number of reports to return      |
| offset     | number | Number of reports to skip        |

**Response:**

```json
{
  "reports": [
    {
      "id": "7c8d65f4-3e22-1a79-99e5-4825d1987f28",
      "name": "Monthly Sales Summary",
      "type": "sales_summary",
      "format": "pdf",
      "created_at": "2023-03-15T18:30:22Z",
      "created_by": "john.smith",
      "status": "completed"
    },
    ...
  ],
  "total": 35,
  "limit": 10,
  "offset": 0
}
```

#### Get Report by ID

```
GET /reports/{report_id}
```

**Response:**

```json
{
  "id": "7c8d65f4-3e22-1a79-99e5-4825d1987f28",
  "name": "Monthly Sales Summary",
  "type": "sales_summary",
  "format": "pdf",
  "created_at": "2023-03-15T18:30:22Z",
  "created_by": "john.smith",
  "status": "completed",
  "parameters": {
    "time_period": "2023-03-01/2023-03-31",
    "departments": ["new", "used", "service"]
  },
  "download_url": "https://storage.watchdogai.com/reports/7c8d65f4-3e22-1a79-99e5-4825d1987f28.pdf"
}
```

#### Schedule a Report

```
POST /reports
```

**Request Body:**

```json
{
  "name": "Weekly Inventory Health Report",
  "type": "inventory_health",
  "format": "pdf",
  "schedule": {
    "frequency": "weekly",
    "day_of_week": "monday",
    "time": "01:00"
  },
  "delivery": {
    "method": "email",
    "recipients": ["john@example.com", "jane@example.com"]
  },
  "parameters": {
    "include_charts": true,
    "include_recommendations": true
  }
}
```

**Response:**

```json
{
  "id": "9d8e72f3-4c32-2b68-77d7-3736d2898e17",
  "name": "Weekly Inventory Health Report",
  "type": "inventory_health",
  "format": "pdf",
  "created_at": "2023-04-15T10:45:33Z",
  "created_by": "current_user",
  "status": "scheduled",
  "next_run": "2023-04-17T01:00:00Z"
}
```

### Data Management

#### Upload Data

```
POST /data/upload
```

**Request:**
Multipart form data with file

**Response:**

```json
{
  "upload_id": "5e6f73d2-1c42-3b57-66d5-2625d7767f16",
  "filename": "march_sales_data.csv",
  "rows": 1250,
  "columns": 24,
  "validation": {
    "status": "success",
    "issues": []
  },
  "timestamp": "2023-04-15T11:30:45Z"
}
```

#### Get Data Uploads

```
GET /data/uploads
```

**Response:**

```json
{
  "uploads": [
    {
      "upload_id": "5e6f73d2-1c42-3b57-66d5-2625d7767f16",
      "filename": "march_sales_data.csv",
      "rows": 1250,
      "columns": 24,
      "uploaded_at": "2023-04-15T11:30:45Z",
      "uploaded_by": "john.smith"
    },
    ...
  ]
}
```

### Audit Logs

#### Get Audit Logs

```
GET /audit/logs
```

**Query Parameters:**

| Name          | Type   | Description                               |
|---------------|--------|-------------------------------------------|
| start_date    | string | Start date (ISO format)                   |
| end_date      | string | End date (ISO format)                     |
| event_type    | string | Filter by event type                      |
| user_id       | string | Filter by user                            |
| resource_type | string | Filter by resource type                   |
| status        | string | Filter by status (success, error, etc.)   |
| limit         | number | Number of logs to return                  |
| offset        | number | Number of logs to skip                    |

**Response:**

```json
{
  "logs": [
    {
      "event": "file_upload",
      "user_id": "john.smith",
      "timestamp": "2023-04-15T11:30:45Z",
      "ip_address": "192.168.1.1",
      "resource_type": "file",
      "resource_id": "5e6f73d2-1c42-3b57-66d5-2625d7767f16",
      "status": "success",
      "details": {
        "filename": "march_sales_data.csv",
        "size": 125000,
        "rows": 1250
      }
    },
    ...
  ],
  "total": 2450,
  "limit": 50,
  "offset": 0
}
```

## Rate Limiting

The API enforces rate limiting to ensure fair usage. The current limits are:

- 100 requests per minute per API key
- 5,000 requests per day per API key

Rate limit headers are included in all API responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1682234567
```

## Error Handling

The API uses standard HTTP response codes to indicate the success or failure of a request.

- 2xx: Success
- 4xx: Client error
- 5xx: Server error

Error responses include a JSON object with details:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid or missing parameters",
    "details": {
      "field": "prompt",
      "issue": "required field is missing"
    }
  }
}
```

## Versioning

The API uses URL versioning (e.g., `/v1/insights`). We maintain backward compatibility within a major version and provide migration guides for major version upgrades.

## SDK Libraries

To simplify API integration, we provide SDK libraries for common programming languages:

- [Python SDK](https://github.com/watchdogai/python-sdk)
- [JavaScript SDK](https://github.com/watchdogai/js-sdk)

## Support

For API support and questions, contact:

- Email: api-support@watchdogai.com
- Documentation: https://docs.watchdogai.com/api