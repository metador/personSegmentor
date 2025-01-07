FROM python:3.11-slim

EXPOSE 5000 

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app/pythonApps/personSegmenter
COPY . .



# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install setuptools in the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
#RUN /app/venv/bin/python3 -m ensurepip
RUN pip3 install -U pip
# Install Python dependencies
COPY ./requirements.txt /app/pythonApps/personSegmenter/
RUN pip3 install --no-cache-dir -r /app/pythonApps/personSegmenter/requirements.txt
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu


ENV PATH="/app/venv/bin:$PATH"

# 
# script to run Flask apps
CMD ["python","/app/pythonApps/personSegmenter/app.py"]
