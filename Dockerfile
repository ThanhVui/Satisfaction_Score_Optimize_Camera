FROM python:3.10

WORKDIR /app
COPY . /app

# Avoid interactive install and fix apt errors
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
