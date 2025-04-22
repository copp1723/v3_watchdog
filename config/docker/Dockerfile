FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY assets/ ./assets/
COPY docs/ ./docs/
COPY profiles/ ./profiles/
COPY prompt_templates/ ./prompt_templates/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Set entry point
ENTRYPOINT ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.headless=true"]