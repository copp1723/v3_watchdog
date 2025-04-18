output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "redis_endpoint" {
  description = "The address of the Redis endpoint"
  value       = module.redis.redis_endpoint
}

output "redis_port" {
  description = "The port of the Redis endpoint"
  value       = module.redis.redis_port
}

output "redis_snapshot_retention" {
  description = "The snapshot retention period for Redis"
  value       = module.redis.snapshot_retention_period
}

output "audit_logs_bucket" {
  description = "The name of the audit logs bucket"
  value       = module.s3.audit_logs_bucket_name
}

output "audit_logs_lifecycle" {
  description = "Lifecycle configuration for audit logs"
  value       = module.s3.audit_logs_lifecycle_rules
}

output "app_data_bucket" {
  description = "The name of the application data bucket"
  value       = module.s3.app_data_bucket_name
}