# start with python 3.9 slim image
FROM python:3.9-slim

# make sure python output shows up in logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# set working directory
WORKDIR /app

# copy requirements first so we can cache the pip install step
COPY requirements.txt .

# install all the python packages we need
RUN pip install --no-cache-dir -r requirements.txt

# copy our training script
COPY train.py .

# create folder for saving the trained model
RUN mkdir -p /app/models

# run the training script when container starts
CMD ["python", "train.py"]

