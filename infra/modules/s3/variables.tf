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