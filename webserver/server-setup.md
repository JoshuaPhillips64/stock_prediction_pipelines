# EC2 Setup for Webserver

This guide provides step-by-step instructions to set up Docker, Docker Compose, clone a GitHub repository, and run Docker Compose on an Amazon Linux 2 EC2 instance.

Once the EC2 Instance is running and Inbound Rules are set correctly. Can SSH in and run below.

### 1. Configure Security Groups (Optional)

Ensure that your EC2 instance's security group allows inbound traffic on:

- SSH | TCP	22 | Your IP	
- HTTP | TCP 80 | 0.0.0.0/0	
- HTTPS	TCP	443	0.0.0.0/0	
- Custom TCP | TCP	7237 | 172.17.0.0/16	Internal Docker network
- Redis	TCP	6379	172.17.0.0/16	Internal Docker network

To configure the security group:
1. Go to the EC2 dashboard in the AWS Console.
2. Find your instance and open its security group settings.

### 2. Update EC2 and Install Docker

First, update the package list and install Docker.

```bash
# Update package list
sudo yum update -y

# Install Docker
sudo yum install docker -y

# Start Docker service
sudo service docker start

# Add the EC2 user to the docker group (avoids using sudo with docker)
sudo usermod -aG docker ec2-user

# Apply the new group membership (you may need to log out and back in for this to take effect)
newgrp docker
```

### 3. Install Docker Compose

Next, install Docker Compose, which allows you to run multi-container Docker applications.

```bash
# Download Docker Compose binary (adjust version if necessary)
sudo curl -L "https://github.com/docker/compose/releases/download/v2.10.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make Docker Compose executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

### 3. Clone the GitHub Repository

Clone the required repository from GitHub.

```bash
# Install git (if not already installed)
sudo yum install git -y

# Clone the stock_prediction_pipelines repository
git clone https://github.com/JoshuaPhillips64/stock_prediction_pipelines.git

# Navigate to the Airflow folder inside the cloned repository
cd stock_prediction_pipelines/src/airflow
```

### 4. Copy the `.env` File

Copy the `.env` file to the Airflow folder.

```bash
nano .env
```

Add the Required Environment Variables . Save and Exit (Ctrl + O, Enter, Ctrl + X).

###



