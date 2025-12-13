# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
# Set the working directory
WORKDIR /code
# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/
# Install dependencies using uv
RUN uv sync --frozen
# Install pip and download spaCy language model for word embeddings
RUN uv pip install pip && uv run python -m spacy download en_core_web_lg
# Copy the application code
COPY ./app /code/app

# Command to run the application on internal port 80
# CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]

# Command to run the application on docker container internal port 8000
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "8000"]