
provider "aws" {
  region     = var.AWS_REGION
  access_key = var.AWS_ACCESS_KEY_ID
  secret_key = var.AWS_SECRET_ACCESS_KEY
}

# Define the key pair for SSH access
resource "aws_key_pair" "deployer_key" {
  key_name   = "deployer_key"
  public_key = file("~/.ssh/id_rsa.pub")  # Adjust to your actual public key location
}

# Define security group for EC2
resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create an EC2 instance t2.2xlarge
resource "aws_instance" "app_server" {
  ami           = "ami-0c55b159cbfafe1f0"  # Ubuntu AMI (adjust based on your region)
  instance_type = "t2.2xlarge"             # Instance type
  key_name      = aws_key_pair.deployer_key.key_name

  # Attach security group for SSH
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]

  # Pass AWS credentials to the EC2 instance via userdata
  user_data = <<-EOF
              #!/bin/bash
              echo "export AWS_ACCESS_KEY_ID=${var.AWS_ACCESS_KEY_ID}" >> /etc/environment
              echo "export AWS_SECRET_ACCESS_KEY=${var.AWS_SECRET_ACCESS_KEY}" >> /etc/environment
              EOF

  tags = {
    Name = "EC2-t2.2xlarge-Model"
  }

  # Output the public IP of the instance
  provisioner "local-exec" {
    command = "echo ${self.public_ip} > ec2_public_ip.txt"
  }
}

output "ec2_public_ip" {
  value = aws_instance.app_server.public_ip
}
