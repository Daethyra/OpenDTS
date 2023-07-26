# Use multi-stage build
# First stage: Use the official Python image as the base image
FROM python:3.11 as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /app

# Update the package list, upgrade all packages and remove unnecessary packages
RUN apt update && apt full-upgrade -y && apt autoremove -y

# Cache dependencies
# Copy only the requirements.txt first, to cache the dependencies
COPY requirements.txt /app/requirements.txt

# Update pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Second stage: Create the final image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the dependencies from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy the current directory contents into the container at /app
COPY . /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' sentinel
USER sentinel
RUN chown -R sentinel:sentinel /app/data

# Make port 443 available to the world outside this container
EXPOSE 443

# Add a healthcheck
HEALTHCHECK CMD python -c 'import sys; sys.exit(0)' || exit 1

# Run app.py when the container launches
CMD ["python","app.py"]
