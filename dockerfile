FROM python:3.9.2-buster

WORKDIR /opt/slam

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install requirements.txt

CMD ["python", "src/slam.py"]