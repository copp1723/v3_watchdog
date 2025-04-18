output "audit_logs_bucket_name" {
  description = "The name of the audit logs bucket"
  value       = aws_s3_bucket.audit_logs.bucket
}

output "audit_logs_bucket_arn" {
  description = "The ARN of the audit logs bucket"
  value       = aws_s3_bucket.audit_logs.arn
}

output "audit_logs_lifecycle_rules" {
  description = "Summary of the lifecycle rules applied to the audit logs bucket"
  value = {
    glacier_transition_days = 30
    expiration_days         = 365
  }
}

output "app_data_bucket_name" {
  description = "The name of the application data bucket"
  value       = aws_s3_bucket.app_data.bucket
}

output "app_data_bucket_arn" {
  description = "The ARN of the application data bucket"
  value       = aws_s3_bucket.app_data.arn
}