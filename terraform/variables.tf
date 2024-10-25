variable "aws_region" {
  description = "La région AWS dans laquelle déployer les ressources"
  type        = string
  default     = "us-east-1"
}

variable "notebook_instance_name" {
  description = "Nom de l'instance SageMaker Notebook"
  type        = string
  default     = "deep-learning-notebook-instance-01"
}

variable "role_name" {
  description = "Nom du rôle IAM pour SageMaker"
  type        = string
  default     = "LabRole"
}
variable "role_arn" {
  description = "L'ARN du rôle IAM pour SageMaker"
  type        = string
}

variable "subnet_id" {
  description = "ID du sous-réseau pour l'instance SageMaker Notebook"
  type        = string
  default     = "subnet-0436bfef25f0eccc8"
}


variable "security_group_ids" {
  description = "ID du groupe de sécurité pour l'instance SageMaker Notebook"
  type        = list(string)
  default     = ["sg-062cc46571c09bb4c"]
}


# variable "bucket_names" {
#   type    = list(string)
#   default = [
#     "dsti-a23-deep-learning-outputs",
#     "backend-terraform-a23dsti-deep-learning-project",
#     "images-projet-deep-learning"
#   ]
# }
