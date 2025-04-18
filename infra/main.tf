/**
 * # V3 Watchdog AI Infrastructure
 *
 * Main Terraform configuration for the V3 Watchdog AI platform.
 * This includes:
 * - VPC and networking
 * - ElastiCache Redis cluster with automatic snapshots
 * - S3 buckets with appropriate lifecycle policies for compliance
 * - RDS PostgreSQL database
 */

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Create VPC for application resources
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "${var.project_name}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway = true
  single_nat_gateway = var.environment != "prod"

  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"
  }
}

# Create Redis ElastiCache cluster
module "redis" {
  source = "./modules/redis"

  project_name    = var.project_name
  environment     = var.environment
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  allowed_cidr_blocks = [var.vpc_cidr]
  
  # Select appropriate instance type based on environment
  node_type       = var.environment == "prod" ? "cache.m5.large" : "cache.t3.small"
  
  # Configure snapshot retention (7 days)
  # This is set in the module
}

# Create S3 buckets
module "s3" {
  source = "./modules/s3"

  project_name = var.project_name
  environment  = var.environment
}

# Security group for application
resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-${var.environment}-app-sg"
  description = "Security group for application servers"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "HTTPS from public"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP from public"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-app-sg"
  }
}