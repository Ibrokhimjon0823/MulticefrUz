#!/bin/bash

# Exit on error
set -e

echo "Waiting for PostgreSQL to start..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 1
done
echo "PostgreSQL started."

# Run migrations
echo "Applying database migrations..."
python manage.py makemigrations --noinput
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start the application
echo "Starting Django application..."
exec "$@"
