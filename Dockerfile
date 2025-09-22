# Use a multi-architecture base image with Selenium pre-installed.
FROM seleniarm/standalone-chromium:latest

# Switch to the root user to gain the necessary permissions for package installation.
USER root

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Return to the default non-root user for security.
USER seluser

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy your application code
COPY . .

# Run the command to start your application
CMD ["python3", "scraper.py"]