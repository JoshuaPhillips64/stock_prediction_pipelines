# EC2 Setup for Airflow

This guide provides step-by-step instructions to set up Docker, Docker Compose, clone a GitHub repository, and run Docker Compose on an Amazon Linux 2 EC2 instance.

Once the EC2 Instance is running and Inbound Rules are set correctly. Can SSH in and run below.

### 1. Configure Security Groups (Optional)

Ensure that your EC2 instance's security group allows inbound traffic on:
- **Port 8080** for Airflow access.
- **Port 22** for SSH access (if required).

To configure the security group:
1. Go to the EC2 dashboard in the AWS Console.
2. Find your instance and open its security group settings.
3. Add an inbound rule to allow HTTP traffic on port `8080` (for Airflow) and SSH on port `22`.

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

### 4. Build and Run Docker Compose

Now, build and start the Docker containers using Docker Compose.

```bash
#Ensure logs is created and has correct permissions
mkdir -p ./logs
sudo chmod -R 777 ./logs
sudo chmod -R 777 ./dags
sudo chmod -R 777 ./scheduler
sudo chmod -R 777 ./plugins

# Build Docker images
docker-compose build

# Start the containers in detached mode
docker-compose up -d
```

### 5. Access Airflow

Once the containers are running, you can access the Airflow webserver by opening your browser and navigating to the public IP address of your EC2 instance on port 8080.

```
http://<your-ec2-public-ip>:8080
```

### Deploymnet Commands:

```bash
#Connect to Airflow container
ssh -i "XXX.pem" XXX.us-east-2.compute.amazonaws.com

#Navigate to the Airflow folder inside the cloned repository
cd stock_prediction_pipelines/src/airflow

#Pull the latest changes from the repository
git pull origin main

docker-compose build

docker-compose down

docker-compose up -d

docker-compose logs -f
```

By following these steps, you’ll have Docker, Docker Compose, and Airflow running on an Amazon Linux 2 EC2 instance, with the stock prediction pipelines set up from the GitHub repository.