variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "watchdog"
}

variable "environment" {
  description = "The deployment environment (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "node_type" {
  description = "The compute and memory capacity of the nodes"
  type        = string
  default     = "cache.t3.small"
}

variable "num_cache_nodes" {
  description = "The number of cache nodes"
  type        = number
  default     = 1
}

variable "engine_version" {
  description = "The version number of the Redis engine"
  type        = string
  default     = "6.2"
}

variable "parameter_group_family" {
  description = "The ElastiCache parameter group family"
  type        = string
  default     = "redis6.x"
}

variable "subnet_ids" {
  description = "The subnet IDs where the Redis cluster will be deployed"
  type        = list(string)
}

variable "vpc_id" {
  description = "The VPC ID where the Redis cluster will be deployed"
  type        = string
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks that are allowed to access the Redis cluster"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}