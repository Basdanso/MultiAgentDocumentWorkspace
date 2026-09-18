
# Generative AI and ML Capstone Project

This project aims to build a Generative AI-powered agent-based knowledge and decision support system. The application allows users to upload documents in multiple formats (.pdf, .txt, .csv, .xlsx, .docx,).
The system uses Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and AgenticAI frameworks to retrieve relevant information, reason over it, and generate accurate, context-aware responses.
AI agents are used to plan, retrieve, reason, and validate the final output.


## Features

- User Interface (UI)
- Real time data processing
- Vector based knowledge store
- Retrival-Augumented Generation pipeline


## Installation

Install my-project with pip

```bash
# Clone the repository
  git clone https://github.com/Basdanso/MultiAgentDocumentWorkspace.git

# Navigate to project directory
  cd /Users/user/PycharmProjects/CAPSTONE_ML_GEN_AI

# Create and activate virtual environment
  python -m venv venv
  venv\Scripts\activate

# Install dependencies
  pip install -r requirements.txt
```

## 🐳 Running with Docker

You can run this application instantly using Docker without needing to install Python or any project dependencies locally.

### Prerequisites
* Ensure you have [Docker Desktop](https://docker.com) installed and running.
* You will need an **OpenAI API Key**.

### 1. Pull the Image from Docker Hub
Download the latest pre-built image to your local machine:
```bash
docker pull basdanso/capstone_ml_gen_ai_flask:latest
```

### 2. Run the Container
Choose one of the two methods below to supply your API key and launch the application:

#### Option A: Pass the key directly in the terminal
```bash
docker run -d -p 5000:5000 -e OPENAI_API_KEY="your_actual_openai_api_key_here" basdanso/capstone_ml_gen_ai_flask:latest
```

#### Option B: Use a local `.env` file
If you already have a `.env` file containing `OPENAI_API_KEY=your_key` in your current directory, run:
```bash
docker run -d -p 5000:5000 --env-file .env basdanso/capstone_ml_gen_ai:latest
```

### 3. Access the Application
Once the container status is active, open your web browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

### 🛑 Stopping the Application
To stop the background container, find its ID and stop it:
```bash
docker ps
docker stop <CONTAINER_ID>
```

## Tech Stack

- Python, Flask, JS, HTML5, CSS, OpenAI, LangChain, LanGraph, Docker


## Contributing

Contributions are always welcome!. Please open an issue or submit a pull request for any changes.

Please adhere to this project's `code of conduct`.

