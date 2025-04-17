
import uuid
import traceback
from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Optional

from RAGsystem import RAGSystem
from text_agent import TextGenerationService
from image_agent import PropertyIssueDetectionAgent
from agent_router import AgentRouter

app = Flask(__name__)
chat_sessions = {}  # Global dictionary to store chat sessions

class ChatSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.chat_history = []  # List to store message objects
        self.rag_system = RAGSystem()
        self.text_gen_service = TextGenerationService()
        self.image_agent = PropertyIssueDetectionAgent()
        self.agent_router = AgentRouter(text_generation_service=self.text_gen_service)
        
        # Add initial system message
        self.add_message(SystemMessage(content="I am an AI assistant that helps with property-related queries."))
        print(f"Created new chat session: {self.session_id}")
        print(f"Initial chat history length: {len(self.chat_history)}")

    def add_message(self, message: BaseMessage):
        """Add a message to the chat history"""
        self.chat_history.append(message)
        print(f"Added message to history. New length: {len(self.chat_history)}")
        print(f"Message content: {message.content[:100]}...")  # Print first 100 chars

    def add_user_message(self, content: str):
        """Add a user message to the chat history"""
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Add an AI message to the chat history"""
        self.add_message(AIMessage(content=content))

    def get_chat_history(self) -> List[BaseMessage]:
        """Get the full chat history"""
        print(f"Retrieving chat history. Length: {len(self.chat_history)}")
        for idx, msg in enumerate(self.chat_history):
            print(f"Message {idx}: {type(msg).__name__} - {msg.content[:50]}...")
        return self.chat_history

    def process_message(self, message_content: str, image_data: Optional[str] = None):
        try:
            # First, add the user's message to history
            self.add_message(HumanMessage(content=message_content))

            # If there's an image, process it and use image analysis as context
            if image_data:
                image_analysis = self.image_agent.analyze_image(image_data, message_content)
                
                # Create a combined message with image caption and detected issues
                image_context = f"Image Analysis:\n{image_analysis['description']}"
                if image_analysis['detected_issues']:
                    issues_text = "\nDetected Issues:\n" + "\n".join(
                        [f"- {issue['issue']}: {issue['description']} (Severity: {issue['severity']})" 
                         for issue in image_analysis['detected_issues']]
                    )
                    image_context += issues_text
                
                # Generate response using image analysis as context
                response = self.text_gen_service.generate_response(
                    user_message=message_content,
                    chat_history=self.chat_history,
                    context=image_context
                )

            # If no image, use RAG system to get relevant context
            else:
                context = self.rag_system.get_relevant_context(message_content)
                print(f"Retrieved context: {context[:200]}...")  # Log first 200 chars
                
                response = self.text_gen_service.generate_response(
                    user_message=message_content,
                    chat_history=self.chat_history,
                    context=context
                )

            # Add the AI response to history
            self.add_message(AIMessage(content=response))

            return {
                "response": response,
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






