# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Update the system packages
RUN apt update && apt full-upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Command to run when the container starts
CMD ["python", "main.py"]
