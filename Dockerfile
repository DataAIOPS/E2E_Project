# Use an official python image as parent image
FROM python:3.9

#Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y bash

# Copy requirements.txt
COPY requirements.txt ./

# RUN requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all content inside docker image
COPY . .

# External IPs can connect to this port
EXPOSE 8000

# RUN the application
CMD ["python", "app.py"]

