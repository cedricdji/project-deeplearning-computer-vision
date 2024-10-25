output "notebook_instance_arn" {
  description = "ARN de l'instance SageMaker Notebook"
  value       = aws_sagemaker_notebook_instance.notebook.arn
}

# output "bucket_names" {
#   description = "Noms des buckets S3 créés"
#   value       = [for bucket in aws_s3_bucket.project_buckets : bucket.bucket]
# }
