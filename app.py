
import uuid
import traceback
from typing import Optional
from flask import Flask, request, jsonify
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from RAGsystem import RAGSystem
from text_agent import TextGenerationService
from image_agent import PropertyIssueDetectionAgent
from agent_router import AgentRouter

app = Flask(__name__)
chat_sessions = {}  # Global dictionary to store chat sessions

class ChatSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.chat_history = []
        
        # Initialize components
        rag_system = RAGSystem()
        text_gen_service = TextGenerationService()
        image_agent = PropertyIssueDetectionAgent()
        
        # Initialize router with all components
        self.agent_router = AgentRouter(
            text_generation_service=text_gen_service,
            image_agent=image_agent,
            rag_system=rag_system
        )
        
        # Add initial system message
        self.add_message(SystemMessage(content="I am an AI assistant that helps with property-related queries."))

    def add_message(self, message: BaseMessage):
        """Add a message to the chat history"""
        self.chat_history.append(message)

    def process_message(self, message_content: str, image_data: Optional[str] = None):
        """Process incoming message and return response"""
        try:
            # Add user message to history
            self.add_message(HumanMessage(content=message_content))

            # Route message and get response
            result = self.agent_router.route_message(
                message=message_content,
                chat_history=self.chat_history,
                image_data=image_data
            )

            # Add AI response to history
            self.add_message(AIMessage(content=result["response"]))

            return {
                "response": result["response"],
                "session_id": self.session_id
            }

        except Exception as e:
            traceback.print_exc()
            error_message = f"Error processing message: {str(e)}"
            print(error_message)
            return {
                "error": error_message,
                "session_id": self.session_id
            }

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id')
        image_data = data.get('image')
        
        # Get or create session
        if session_id and session_id in chat_sessions:
            session = chat_sessions[session_id]
        else:
            session = ChatSession()
            chat_sessions[session.session_id] = session
            session_id = session.session_id

        # Process message and get response
        response = session.process_message(message, image_data)

        return jsonify({
            'response': response.get('response'),
            'session_id': session.session_id
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)








