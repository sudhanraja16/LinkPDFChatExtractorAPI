# 🚀 FastAPI Text Processing Application

This application provides a REST API for processing text from PDFs and URLs using embeddings for intelligent text retrieval. It uses the BGE small model for generating embeddings and FAISS for efficient similarity search.

## ✨ Features

- 📄 Extract and process text from PDF files
- 🔗 Scrape and process text from URLs
- 🧠 Generate embeddings using BGE small model
- 🔍 Perform similarity search on processed text
- 🐳 Docker support for easy deployment

## 📋 Prerequisites

- 🐳 Docker
- 🔧 Docker Compose

The application uses the following main dependencies:
- ⚡ FastAPI
- 📑 PyMuPDF (for PDF processing)
- 🕷️ BeautifulSoup4 (for URL scraping)
- 🔎 FAISS (for similarity search)
- 🤗 HuggingFace BGE embeddings

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/sudhanraja16/LinkPDFChatExtractorAPI.git
cd LinkPDFChatExtractorAPI
```

2. Ensure you have the BGE small model in your project directory:
```
bge-small-en/
```

3. Build and run the Docker container:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## 🔌 API Endpoints

### 1. 📄 Process PDF
- **Endpoint**: `/process_pdf`
- **Method**: POST
- **Description**: Upload and process a PDF file
- **Request**: Multipart form data with PDF file
- **Response**: JSON containing chat ID and success message

Example using curl:
```bash
curl -X POST -F "file=@path/to/your/file.pdf" http://localhost:8000/process_pdf
```

### 2. 🔗 Process URL
- **Endpoint**: `/process_url`
- **Method**: POST
- **Description**: Process text from a given URL
- **Request**: JSON containing URL
- **Response**: JSON containing chat ID and success message

Example using curl:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url":"https://example.com"}' \
     http://localhost:8000/process_url
```

### 3. 💬 Chat
- **Endpoint**: `/chat`
- **Method**: POST
- **Description**: Query processed text using natural language
- **Request**: JSON containing chat ID and query
- **Response**: JSON containing relevant text chunk

Example using curl:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"chat_id":"your-chat-id","query":"your query here"}' \
     http://localhost:8000/chat
```

## 📁 Project Structure

```
project-root/
│
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── bge-small-en/         # BGE model directory
└── README.md             # This file
```

## 👩‍💻 Development

To run the application in development mode:

1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app:app --reload
```

## 🐳 Docker Commands

- Build and start the container:
```bash
docker-compose up --build
```

- Start the container in detached mode:
```bash
docker-compose up -d
```

- Stop the container:
```bash
docker-compose down
```

## ⚠️ Troubleshooting

1. If you encounter memory issues with FAISS, you may need to adjust the Docker container's memory limit in `docker-compose.yml`.

2. Ensure that the `bge-small-en` model is correctly placed in your project directory before building the Docker image.

## 📝 License

[Your License Here]

---

<div align="center">
  
### 🛠️ Built With

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>
