# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Prevent Python from buffering stdout/stderr (better logs)
ENV PYTHONUNBUFFERED=1

# 3. Install system dependencies required for OpenCV
# This fixes the "libGL.so.1: cannot open shared object file" error
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy requirements first (to cache dependencies)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application code
COPY . .

# 8. Create necessary folders so the app doesn't crash on permission errors
RUN mkdir -p uploads outputs

# 9. Run the app using Gunicorn
# Railway assigns a random PORT, so we must bind to 0.0.0.0:$PORT
CMD gunicorn app:app -b 0.0.0.0:$PORT