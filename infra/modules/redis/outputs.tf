output "redis_endpoint" {
  description = "The address of the Redis endpoint"
  value       = aws_elasticache_cluster.watchdog_redis.cache_nodes[0].address
}

output "redis_port" {
  description = "The port of the Redis endpoint"
  value       = aws_elasticache_cluster.watchdog_redis.cache_nodes[0].port
}

output "redis_sg_id" {
  description = "The ID of the Redis security group"
  value       = aws_security_group.redis_sg.id
}

output "redis_cluster_id" {
  description = "The ID of the Redis cluster"
  value       = aws_elasticache_cluster.watchdog_redis.id
}

output "snapshot_retention_period" {
  description = "The number of days for which ElastiCache will retain automatic snapshots"
  value       = aws_elasticache_cluster.watchdog_redis.snapshot_retention_limit
}