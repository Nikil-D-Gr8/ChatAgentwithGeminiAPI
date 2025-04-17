from typing import List, Optional
from langchain_core.messages import BaseMessage
import google.generativeai as genai
import os

class TextGenerationService:
    def __init__(self):
        print("Initializing Text Generation Service...")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self, user_message: str, chat_history: List[BaseMessage], context: Optional[str] = None) -> str:
        """Generate a response based on the user message, chat history, and retrieved context"""
        try:
            # Format chat history into a string
            history_str = "\n".join([
                f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
                for msg in chat_history[:-1]  # Exclude the latest user message as we'll add it separately
            ])

            # Construct the prompt with all available information
            prompt = f"""
Context information: {context if context else 'No additional context available'}

Previous conversation:
{history_str}

Current user message: {user_message}

Please provide a helpful response based on the above information.
"""
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            return response.text

        except Exception as e:
            print(f"Error generating text response: {e}")
            raise







