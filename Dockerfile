# 1. Use official Python image
FROM python:3.10-slim-buster

# 2. Set working directory
WORKDIR /app

# 3. Copy the app source code
COPY . .

# 4. Install dependencies

RUN pip install --no-cache-dir -r requirements.txt



# 5. Expose port
EXPOSE 8080

# 6. Start FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]