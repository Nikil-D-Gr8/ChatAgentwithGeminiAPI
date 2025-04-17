
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

    def generate_response(self, message: str, image_data: Optional[str] = None) -> str:
        """Generate a response using the chat history"""
        try:
            print(f"\nGenerating response for session {self.session_id}")
            print(f"Current chat history length: {len(self.chat_history)}")
            
            # Add user message to history
            self.add_message(HumanMessage(content=message))
            
            if image_data:
                # Handle image analysis
                analysis = self.image_agent.analyze_image(
                    image_data=image_data,
                    user_query=message
                )
                
                response_text = f"I see {analysis['description']}. "
                if analysis['detected_issues']:
                    response_text += "\n\nI've detected the following issues:\n"
                    for issue in analysis['detected_issues']:
                        response_text += f"- {issue['issue']} (Severity: {issue['severity']}): {issue['description']}\n"
                
                self.add_message(AIMessage(content=response_text))
                return response_text
            else:
                # Get relevant context from RAG system
                relevant_context = self.rag_system.get_relevant_context(message)
                print(f"Retrieved relevant context: {relevant_context[:200]}...")  # Print first 200 chars for debugging
                
                # Format chat history for the text generation service
                formatted_history = [
                    {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                     "content": msg.content}
                    for msg in self.chat_history[:-1]  # Exclude the last message as it's the current one
                ]
                
                # Generate response with context
                response = self.text_gen_service.generate_response(
                    message=message,
                    chat_history=formatted_history,
                    context=relevant_context
                )
                
                self.add_message(AIMessage(content=response))
                return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        session_id = data.get('session_id')
        image_data = data.get('image')

        # Get or create session
        if session_id and session_id in chat_sessions:
            print(f"Using existing session: {session_id}")
            session = chat_sessions[session_id]
        else:
            print(f"Creating new session (old session_id was: {session_id})")
            session = ChatSession()
            chat_sessions[session.session_id] = session
            print(f"New session created with ID: {session.session_id}")

        # Generate response
        response = session.generate_response(message, image_data)

        return jsonify({
            'response': response,
            'session_id': session.session_id
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_session():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id and session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({'status': 'success', 'message': 'Session reset successfully'})
    
    return jsonify({'status': 'error', 'message': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
















