/**
 * # S3 Bucket Module
 *
 * Creates an AWS S3 bucket with the following features:
 * - Encryption at rest
 * - Versioning (optional)
 * - Lifecycle rules for data transition and expiration
 * - Audit log bucket with compliance-ready retention policies
 */

resource "aws_s3_bucket" "audit_logs" {
  bucket = "${var.project_name}-${var.environment}-audit-logs"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-audit-logs"
    Environment = var.environment
    Project     = var.project_name
    Terraform   = "true"
    DataType    = "compliance"
  }
}

resource "aws_s3_bucket_acl" "audit_logs_acl" {
  bucket = aws_s3_bucket.audit_logs.id
  acl    = "private"
}

resource "aws_s3_bucket_versioning" "audit_logs_versioning" {
  bucket = aws_s3_bucket.audit_logs.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "audit_logs_encryption" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "audit_logs_lifecycle" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    id      = "audit-log-retention"
    status  = "Enabled"
    
    # Transition to Glacier after 30 days
    transition {
      days          = 30
      storage_class = "GLACIER"
    }
    
    # Expire objects after 365 days (1 year)
    expiration {
      days = 365
    }
  }
}

resource "aws_s3_bucket_public_access_block" "audit_logs_access" {
  bucket = aws_s3_bucket.audit_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Additional bucket for application data
resource "aws_s3_bucket" "app_data" {
  bucket = "${var.project_name}-${var.environment}-app-data"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-app-data"
    Environment = var.environment
    Project     = var.project_name
    Terraform   = "true"
    DataType    = "application"
  }
}

resource "aws_s3_bucket_acl" "app_data_acl" {
  bucket = aws_s3_bucket.app_data.id
  acl    = "private"
}

resource "aws_s3_bucket_versioning" "app_data_versioning" {
  bucket = aws_s3_bucket.app_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_data_encryption" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app_data_access" {
  bucket = aws_s3_bucket.app_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}