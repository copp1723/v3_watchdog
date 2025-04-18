#!/bin/bash
# Script to run terraform plan and validate infrastructure changes

# Set environment variables
export TF_VAR_project_name=watchdog
export TF_VAR_environment=dev

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "Terraform could not be found. Please install it first."
    exit 1
fi

# Initialize Terraform if not already done
if [ ! -d ".terraform" ]; then
    echo "Initializing Terraform..."
    terraform init
fi

# Run terraform validate
echo "Running terraform validate..."
terraform validate

if [ $? -ne 0 ]; then
    echo "Terraform validation failed. Please fix the errors above."
    exit 1
fi

# Run terraform plan
echo "Running terraform plan..."
terraform plan -out=tfplan

if [ $? -ne 0 ]; then
    echo "Terraform plan failed. Please fix the errors above."
    exit 1
fi

echo "Terraform plan succeeded and was saved to 'tfplan'."
echo "To apply these changes, run: terraform apply tfplan"

# Show a summary of the changes
echo "Plan Summary:"
echo "=============="
echo "Redis snapshot retention: 7 days"
echo "Audit logs bucket lifecycle configuration:"
echo "  - Transition to Glacier after 30 days"
echo "  - Expire objects after 365 days"