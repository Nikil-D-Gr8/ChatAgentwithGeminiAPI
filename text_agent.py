import traceback
from typing import List, Dict
import google.generativeai as genai

class TextGenerationService:
    def __init__(self):
        print("Initializing Text Generation Service...")
        self.model = genai.GenerativeModel('gemini-2.0-flash') 

    def generate_response(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate response using Gemini with chat history"""
        try:
            print(f"\nGenerating response with chat history length: {len(chat_history)}")
            
            # Start a new chat
            chat = self.model.start_chat(history=[])
            
            # Add all historical messages to the chat
            for msg in chat_history:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
                elif msg["role"] == "assistant":
                    # For assistant messages, we need to add them to the history
                    chat.history.append({"role": "model", "parts": [msg["content"]]})
            
            # Send the current message and get response
            response = chat.send_message(message)
            print(f"Generated response: {response.text[:100]}...")
            
            return response.text

        except Exception as e:
            print(f"Error in TextGenerationService: {str(e)}")
            traceback.print_exc()
            return f"I apologize, but I encountered an error: {str(e)}"


