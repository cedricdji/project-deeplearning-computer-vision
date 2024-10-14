
output "ec2_public_ip" {
  value = aws_instance.app_server.public_ip
}

output "ec2_public_dns" {
  value = aws_instance.app_server.public_dns
}

output "ec2_instance_id" {
  value = aws_instance.app_server.id
}