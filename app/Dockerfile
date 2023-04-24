FROM python:3.10.11-slim
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]