
# variable "AWS_ACCESS_KEY_ID" {
#   description = "Your AWS access key"
#   type        = string
# }

# variable "AWS_SECRET_ACCESS_KEY" {
#   description = "Your AWS secret access key"
#   type        = string
# }

variable "AWS_REGION" {
  description = "The region in which AWS resources are created"
  type        = string
  default     = "us-west-2" # Default region, adjust if necessary
}

# variable "SSH_PUBLIC_KEY" {
#   description = "For SSH access"
#   type        = string
# }

# variable "AWS_SESSION_TOKEN" {
#   description = "AWS Session Token for temporary credentials"
#   type        = string
# }
