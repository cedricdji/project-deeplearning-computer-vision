terraform {
  backend "s3" {
    bucket = "backend-terraform-a23dsti-deep-learning-project"
    key    = "ingeneurie/terraform.tfstate"
    region = "us-east-1"
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}
provider "aws" {
  region = var.aws_region
}

# Création des buckets S3 s'ils n'existent pas
resource "aws_s3_bucket" "project_buckets" {
  for_each = toset(var.bucket_names)
  bucket   = each.value
  acl = "private"
}

# Configuration du cycle de vie du notebook
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook_lifecycle_config" {
  name = "DownloadModelFiles"

  on_start = <<EOF
#!/bin/bash
# Téléchargement du fichier Model.ipynb et requirements.txt dans le répertoire SageMaker

# Chemin de destination
cd /home/ec2-user/SageMaker

# Téléchargement depuis S3
aws s3 cp s3://images-projet-deep-learning/Model.ipynb .
aws s3 cp s3://images-projet-deep-learning/requirements.txt .

echo "Fichiers Model.ipynb et requirements.txt téléchargés dans le répertoire SageMaker"
EOF
}

# Création de l'instance Notebook SageMaker
resource "aws_sagemaker_notebook_instance" "notebook" {
  name                   = var.notebook_instance_name
  instance_type          = "ml.t2.medium"
  role_arn               = var.role_arn
  direct_internet_access = "Enabled"
  root_access            = "Enabled"
  volume_size            = 10
  lifecycle_config_name  = aws_sagemaker_notebook_instance_lifecycle_configuration.notebook_lifecycle_config.name

  tags = {
    Name = "DeepLearningNotebook"
  }
}
