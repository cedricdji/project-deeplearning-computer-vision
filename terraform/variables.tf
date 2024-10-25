variable "aws_region" {
  description = "La région AWS dans laquelle déployer les ressources"
  type        = string
  default     = "us-east-1"
}

variable "notebook_instance_name" {
  description = "Nom de l'instance SageMaker Notebook"
  type        = string
  default     = "deep-learning-notebook-instance"
}

variable "role_arn" {
  description = "ARN du rôle IAM pour l'instance SageMaker Notebook"
  type        = string
}

variable "bucket_names" {
  description = "Liste des noms de buckets S3 à créer pour le projet"
  type        = list(string)
  default     = [
    "dsti-a23-deep-learning-outputs",
    "backend-terraform-a23dsti-deep-learning-project",
    "images-projet-deep-learning"
  ]
}
