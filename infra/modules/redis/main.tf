/**
 * # Redis ElastiCache Module
 *
 * Creates an AWS ElastiCache Redis cluster with the following features:
 * - Automatic daily snapshots with 7-day retention
 * - Encryption at rest
 * - Configurable node type and count
 * - Security group for network access control
 */

resource "aws_elasticache_cluster" "watchdog_redis" {
  cluster_id           = "${var.project_name}-${var.environment}-redis"
  engine               = "redis"
  node_type            = var.node_type
  num_cache_nodes      = var.num_cache_nodes
  parameter_group_name = aws_elasticache_parameter_group.watchdog_redis_params.name
  engine_version       = var.engine_version
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.watchdog_redis_subnet.name
  security_group_ids   = [aws_security_group.redis_sg.id]
  
  # Configure automatic snapshots with 7-day retention
  snapshot_retention_limit = 7
  snapshot_window          = "00:00-01:00"  # Daily snapshot window (UTC)
  
  # Enable encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled  = true
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-redis"
    Environment = var.environment
    Project     = var.project_name
    Terraform   = "true"
  }
}

resource "aws_elasticache_parameter_group" "watchdog_redis_params" {
  name   = "${var.project_name}-${var.environment}-redis-params"
  family = var.parameter_group_family

  parameter {
    name  = "maxmemory-policy"
    value = "volatile-lru"
  }
}

resource "aws_elasticache_subnet_group" "watchdog_redis_subnet" {
  name       = "${var.project_name}-${var.environment}-redis-subnet"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "redis_sg" {
  name        = "${var.project_name}-${var.environment}-redis-sg"
  description = "Security group for Redis ElastiCache cluster"
  vpc_id      = var.vpc_id

  ingress {
    description = "Redis from VPC"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-sg"
    Environment = var.environment
    Project     = var.project_name
    Terraform   = "true"
  }
}