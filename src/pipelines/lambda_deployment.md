### Steps:

1. **Build the Docker Image Locally**  
   Run this command after changing the code and when you want to deploy the updated version:
   ```bash
   docker build --no-cache -t lambda-deployments:latest .
   ```
2**Set AWS Credentials**  
   Ensure you have the correct AWS credentials configured. Run:
   ```bash
   aws configure
   ```
3**Login to AWS ECR**

   - **If AWS CLI is Set Up Correctly:**  
     For standard setups, run:
     ```bash
     aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ECR_REPO_URL>
     ```

2**Tag the Docker Image**  
   Tag the Docker image to prepare it for pushing to AWS ECR:
   ```bash
   # Tag your image
  docker tag lambda-deployments:latest <ECR_REPO_URL>/lambda-deployments:latest
   ```

6. **Push the Docker Image to AWS ECR**  
   Once tagged and logged in, push the image to AWS ECR:
   ```bash
   docker push <ECR_REPO_URL>/lambda-deployments:latest
   ```

7**Set Up the Lambda**  
   Create a new Lambda function in the AWS Console. Set the timeout to 15 minutes and the memory higher.
   Right now, need to manually add the environment variables to the Lambda function under configuration. Only need 10 or so. 
   This will be automated in the future.
   Future goal is to get this into terraform

This process will build, test, tag, and push the Selenium Docker image to AWS Lambda via AWS ECR for deployment.