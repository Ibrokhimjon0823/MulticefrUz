FROM python:3.11.4-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    python3-dev \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Copy entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create directory for audio files
RUN mkdir -p /app/audio_files

# Set entrypoint script
ENTRYPOINT ["sh", "/app/entrypoint.sh"]

# Default command to run the bot
# CMD ["python", "manage.py", "runbot"]
