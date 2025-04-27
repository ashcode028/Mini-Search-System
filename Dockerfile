# Use the official Python 3.12 image as the base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


RUN apt-get update && apt-get install -y curl gcc

# Download and install Miniconda based on architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        curl -sSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        curl -sSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the pip dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install conda dependencies
RUN conda install -c conda-forge faiss-cpu spacy nltk

# Download the spaCy model (en_core_web_sm)
RUN python -m spacy download en_core_web_sm

RUN python -c "import nltk; nltk.download('stopwords')"

# Copy the FastAPI application code
COPY . /app/

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
