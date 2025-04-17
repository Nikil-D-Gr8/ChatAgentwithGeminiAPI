# AI Property Assistant

An intelligent chat assistant that handles both text conversations and property image analysis using Gemini AI.

## Features
- Text-based property queries and conversations
- Image analysis for property issues and features
- Context-aware responses using RAG (Retrieval Augmented Generation)
- Multi-session support
- Persistent chat history

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Project Structure

```
├── app.py              # Session management and API endpoints
├── agent_router.py     # Central routing logic for all agents
├── text_agent.py       # Text generation service using Gemini
├── image_agent.py      # Image analysis service
├── rag_system.py       # Retrieval-Augmented Generation system
└── .env               # Environment variables
```

## Architecture

### Components

1. **ChatSession (app.py)**
   - Manages individual chat sessions
   - Maintains message history
   - Coordinates with AgentRouter

2. **AgentRouter (agent_router.py)**
   - Routes messages to appropriate services
   - Handles both text and image requests
   - Coordinates RAG system integration
   - Manages response generation flow

3. **Text Generation (text_agent.py)**
   - Processes text queries
   - Generates contextual responses
   - Integrates with Gemini AI

4. **Image Analysis (image_agent.py)**
   - Analyzes property images
   - Detects issues and features
   - Provides detailed image descriptions

5. **RAG System (rag_system.py)**
   - Retrieves relevant context
   - Enhances response accuracy
   - Manages knowledge base

## API Reference

### POST /chat
Process new messages and generate responses.

**Request Body:**
```json
{
    "message": "string",
    "session_id": "string (optional)",
    "image_data": "string (optional, base64)"
}
```

**Response:**
```json
{
    "response": "string",
    "session_id": "string"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid request
- 500: Server error



## Development

### Running Locally

```bash
python app.py
```

## Using the Test Client

The `test_client.py` provides a command-line interface for testing the chat API. It supports both text-only conversations and image+text queries.

### Running the Test Client
```bash
python test_client.py
```

### Available Commands
- `quit` - Exit the program
- `reset` - Start a new session
- `image "path/to/image.jpg" Your message here` - Send an image with accompanying text

### Examples

1. Text-only conversation:
```
> How can I improve my property's curb appeal?
AI: [Response about property improvements]
```

2. Sending an image with text:
```
> image "photos/house_front.jpg" What issues do you see with this property?
AI: [Analysis of the property image and response]
```

### Image Requirements
- Supported formats: JPG, PNG
- Path must be in quotes
- Message is required after the image path

### Example Session
```
Chat API Client
Commands:
  'quit' - Exit the program
  'reset' - Start new session
  'image "path" message' - Send message with image (path must be in quotes)
Session ID: None

> image "house.jpg" What's wrong with this house?
AI: [Analysis of house image]

> How much would those repairs cost?
AI: [Response about repair costs]

> reset
Session reset successfully

> quit
```

### Troubleshooting
- If image path is not found: Check that the path is correct and use forward slashes
- If image fails to send: Ensure the image format is supported and file is not corrupted
- If no response: Check that the server is running and accessible

