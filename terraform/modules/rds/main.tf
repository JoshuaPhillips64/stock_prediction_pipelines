# Creates a PostgreSQL RDS instance

resource "aws_db_instance" "postgres" {
  allocated_storage       = 20
  storage_type            = "gp2"
  engine                  = "postgres"
  engine_version          = "13.4"
  instance_class          = "db.t3.micro"
  identifier              = var.db_name
  username                = var.db_username
  password                = var.db_password
  parameter_group_name    = "default.postgres13"
  skip_final_snapshot     = true
  publicly_accessible     = false
  vpc_security_group_ids  = [aws_security_group.rds.id]
  db_subnet_group_name    = aws_db_subnet_group.rds_subnet.id
  multi_az                = false
  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"
  tags = {
    Name        = "${var.environment}-postgresql"
    Environment = var.environment
  }
}

resource "aws_db_subnet_group" "rds_subnet" {
  name       = "${var.environment}-rds-subnet-group"
  subnet_ids = var.private_subnets
  tags = {
    Name        = "${var.environment}-rds-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "rds" {
  name        = "${var.environment}-rds-sg"
  description = "Security group for RDS"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_groups
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.environment}-rds-sg"
    Environment = var.environment
  }
}