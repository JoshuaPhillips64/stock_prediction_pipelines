# Start from Amazon Linux 2
FROM amazon/aws-lambda-python:3.12

WORKDIR /var/task

# Install necessary tools
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y python3-pip python3 python3-setuptools unzip gzip yum-utils tar wget \
    gcc openssl-devel bzip2-devel libffi-devel postgresql-devel && \
    yum clean all

RUN python --version

# Install Pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py

# Install Poetry
RUN pip3 install poetry

RUN yum install -y atk cups-libs gtk3 libXcomposite alsa-lib \
    libXcursor libXdamage libXext libXi libXrandr libXScrnSaver \
    libXtst pango at-spi2-atk libXt xorg-x11-server-Xvfb \
    xorg-x11-xauth dbus-glib dbus-glib-devel nss mesa-libgbm jq unzip

# Copy your pyproject.tol into the Docker container
COPY pyproject.toml ./

# Install and zip dependencies - > output is layer.zip
RUN python3 -m venv /var/task/venv && \
    . /var/task/venv/bin/activate && \
    poetry config virtualenvs.create false && \
    poetry install

# Set the Python path to include the site-packages
ENV PYTHONPATH="/var/task:/var/task/venv/lib/python3.12/site-packages:${PYTHONPATH}"

# Copy the lambda application code
COPY ingest_alpha_vantage/handler.py ./

# Copy the rest of the application functions
COPY *.py ./

# Command to run the Lambda function
CMD ["handler.main" ]