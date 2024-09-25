
variable "AWS_ACCESS_KEY_ID" {
  description = "Your AWS access key"
  type        = string
}

variable "AWS_SECRET_ACCESS_KEY" {
  description = "Your AWS secret access key"
  type        = string
}

variable "AWS_REGION" {
  description = "The region in which AWS resources are created"
  type        = string
  default     = "us-east-1"  # Default region, adjust if necessary
}

variable "SSH_PUBLIC_KEY" {
  description = "For SSH access"
  type        = string 
}