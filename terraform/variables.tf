variable "AWS_ACCESS_KEY_ID" {
  description = "Your AWS access key"
  type        = string
  default    = "" # Default value, adjust if necessary
}

variable "AWS_SECRET_ACCESS_KEY" {
  description = "Your AWS secret access key"
  type        = string
  default     = "" # Default value, adjust if necessary
}

variable "AWS_REGION" {
  description = "The region in which AWS resources are created"
  type        = string
  default     = "us-east-1" # Default region, adjust if necessary
}

variable "SSH_PUBLIC_KEY" {
  description = "For SSH access"
  type        = string
  default     = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCpPJ6/OEvOz2nHwmXd/lycimDUv9u1ZRRbVTbdu8M2zi7Ywji1MGgiVp8t623441kUxXIsuWlJbTCWyL9Q+Ifgwk3oK8ey7JyOob1z3V3ZxSMzN2Kgv3dDnrQk5luxjzkdUOiujAiHwk2yYc+1LX3svGJIzieW52Eegufm8UosKZFrAFb5dVIxVaaoWdg6LNeYh1NqkKSw7z2AtiA/RQfiybOicgLxxD8Ea7kRMebjKQJd4M9NeeDr/ooIlb9swFvcGWhFXqPLmTQeI6VrlVPcOURTuVbJ4esE4AVtXOyU3Zxf0A3fWEZpdl+i/xJHU5HXoL8Ix5Nlwyu4S+WzvFVud6nHVi/moyU8QiDCcf82J83QvCQpExhL/QfDTtblpZ3TKN705kplOMJcBz01KD0iowNmYGCs/iO0X9OsnE//Yyk5Hb4ZpTr4ZtsA2Hd1egH740kK7o38OgowJUa99jEKIusHU3L5FYP3xCkPp965qx+hpr1MMGHRG31JptsiAVqVkp2QWMnkqfU2PzEIO3EemLQkbw/oiuHyNflA63mgH+Heq04n7ljNnfn0/CHh8vT8nYDTYQdIFDdxzbIaphGnxnA7BBgHmf9HijyrWdQx7zu5vOpbZvt2OWkcHBY3eeRIsagiW069wAXbMiYsfb/lktlB2Unc/JBt//ZN+w1xWQ== cedric_dji@yahoo.fr" # Default value, adjust if necessary
}

variable "AWS_SESSION_TOKEN" {
  description = "AWS Session Token for temporary credentials"
  type        = string
  default     = "" # Default value, adjust if necessary
}

variable "AWS_ACCOUNT_ID" {
  description = "Your AWS account ID"
  type        = string
  default     = "" # Default value, adjust if necessary
}
