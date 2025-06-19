FROM python:3.12-slim
RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential curl
RUN apt-get -y install python3-dev python3-pip

COPY . ./
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /app
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app