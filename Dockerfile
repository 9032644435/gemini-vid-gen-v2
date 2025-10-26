# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies (good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
# Add gunicorn here explicitly
RUN pip install --no-cache-dir gunicorn -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the command to run your application using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
