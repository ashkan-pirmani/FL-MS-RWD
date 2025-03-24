# Use the official Miniconda image as a base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the Conda environment file into the container
COPY environment.yml .

# Create the Conda environment specified in environment.yml
RUN conda env create -f environment.yml

# Use the Conda environment for all subsequent commands
SHELL ["conda", "run", "-n", "fl", "/bin/bash", "-c"]

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run your application (update main.py if needed)
CMD ["python", "main.py"]