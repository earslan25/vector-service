# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code to the container's workspace
COPY . .

# Run the command to generate the protos
RUN chmod +x ./generate_protos.sh
RUN ./generate_protos.sh

# Set the command to run your application
CMD ["python", "main.py"]
