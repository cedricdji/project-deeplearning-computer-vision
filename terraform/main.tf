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
# resource "aws_s3_bucket" "project_buckets" {
#   for_each       = toset(var.bucket_names)
#   bucket         = each.value
#   acl            = "private"
#   force_destroy  = true

#   lifecycle {
#     prevent_destroy = true
#     ignore_changes = [bucket]
#   }
# }

# Politique IAM pour autoriser iam:PassRole
# resource "aws_iam_role_policy" "pass_role_policy" {
#   name   = "PassRolePolicy"
#   role   = var.role_name  # Utilisez le rôle défini dans les secrets pour SageMaker

#   policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [
#       {
#         Effect   = "Allow",
#         Action   = "iam:PassRole",
#         Resource = var.role_name  # Autorise l'utilisation du rôle pour SageMaker
#       }
#     ]
#   })
# }

# Configuration du cycle de vie du notebook avec encodage en base64
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook_lifecycle_config" {
  name = "DownloadAndRunModel"

  on_start = base64encode(<<EOF
#!/bin/bash
# Script de démarrage de l'instance SageMaker

cd /home/ec2-user/SageMaker

# Télécharger model.ipynb et requirements.txt depuis S3
aws s3 cp s3://images-projet-deep-learning/model.ipynb .
aws s3 cp s3://images-projet-deep-learning/requirements.txt .

# Installer les dépendances depuis requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Installer papermill pour l'exécution automatique du notebook
pip install papermill

# Exécuter model.ipynb avec papermill et sauvegarder la sortie dans output.ipynb
papermill model.ipynb output.ipynb

echo "Exécution du modèle terminée."
EOF
  )
}


# Création de l'instance Notebook SageMaker
resource "aws_sagemaker_notebook_instance" "notebook" {
  name                   = var.notebook_instance_name
  instance_type          = "ml.t2.medium"
  role_arn               = var.role_arn
  direct_internet_access = "Enabled"
  root_access            = "Enabled"
  volume_size            = 5
# lifecycle_config_name  = aws_sagemaker_notebook_instance_lifecycle_configuration.notebook_lifecycle_config.name
  subnet_id              = var.subnet_id
  security_groups = var.security_group_ids
  tags = {
    Name = "DeepLearningNotebook"

  }
}
