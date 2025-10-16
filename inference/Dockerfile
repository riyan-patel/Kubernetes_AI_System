# start with python 3.9 slim image
FROM python:3.9-slim

# make sure python output shows up in logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# install gcc because some python packages need it to compile
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy requirements first so we can cache the pip install step
COPY requirements.txt .

# install all the python packages we need
RUN pip install --no-cache-dir -r requirements.txt

# copy our api code
COPY app.py .
COPY model_loader.py .

# create folder for the trained model
RUN mkdir -p /app/models

# expose port 8000 for the api
EXPOSE 8000

# start the api server when container starts
CMD ["python", "app.py"]
