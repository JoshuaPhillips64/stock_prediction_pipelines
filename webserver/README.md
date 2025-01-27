# EC2 Setup for Webserver

This guide provides step-by-step instructions to set up Docker, Docker Compose, clone a GitHub repository, and run Docker Compose on an Amazon Linux 2 EC2 instance.

Once the EC2 Instance is running and Inbound Rules are set correctly. Can SSH in and run below.

###Prerequisites
- AWS Account: Access to an AWS account with permissions to create EC2 instances, manage Route 53, and access Certificate Manager.

- Domain Name: is set up in AWS Route 53. Need to add DNS A routes for www. and @ to the EC2 instance's public IP OR Elastic IP.

- SSL Certificate: An SSL certificate exists in AWS Certificate Manager and has been validated for the domain name. 
Note: ACM certificates cannot be directly used with Nginx on EC2 instances. We will obtain SSL certificates using Let's Encrypt.

- SSH Key Pair: An SSH key pair to securely access your EC2 instance.

### 1. Configure Security Groups (Optional)

Ensure that your EC2 instance's security group allows inbound traffic.

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
cd stock_prediction_pipelines/webserver
```

### 4. Copy the `.env` File

Copy the `.env` file to the Airflow folder.

```bash
nano .env
```

Add the Required Environment Variables . Save and Exit (Ctrl + O, Enter, Ctrl + X).

### 5. Install Certbot and Obtain SSL Certificates

```bash
# Install Certbot
sudo yum install -y certbot

# Stop Any Services Using Port 80
sudo systemctl stop nginx

# Obtain SSL Certificates Using Certbot
sudo certbot certonly --standalone -d smartstockpredictor.com -d www.smartstockpredictor.com
```
Certificates Location:

Certificates will be stored in /etc/letsencrypt/live/smartstockpredictor.com/.

### 6. Create a Docker Volume for Certificates

Create a Docker volume to store the SSL certificates.

```bash
# Create a Docker volume for the certificates
docker volume create --name letsencrypt-certs

# Copy the SSL certificates to the Docker volume
docker run --rm \
  -v letsencrypt-certs:/data \
  -v /etc/letsencrypt:/source \
  alpine \
  sh -c "cp -r /source/* /data/"
```

### 6. Start Docker Compose

```bash
# Start Docker Compose
docker-compose up -d --build

# Check the status of the Docker containers
docker-compose ps
```

### Deployment Commands:

```bash
#Connect to Webserver container
ssh -i "XXX.pem" XXX.us-east-2.compute.amazonaws.com

#Navigate to the folder inside the cloned repository
cd stock_prediction_pipelines/webserver

#Pull the latest changes from the repository
git reset --hard origin/main

git pull origin main

docker system prune -f

docker-compose build --no-cache

docker-compose down

docker-compose up -d

docker-compose logs -f
```

### Renew Cert:

```bash
#Connect to Webserver container
ssh -i "XXX.pem" XXX.us-east-2.compute.amazonaws.com

#Navigate to the folder inside the cloned repository
cd stock_prediction_pipelines/webserver

docker-compose down

sudo certbot renew

docker run --rm -v letsencrypt-certs:/data -v /etc/letsencrypt:/source alpine sh -c "cp -r /source/* /data/"

docker-compose up -d
```






