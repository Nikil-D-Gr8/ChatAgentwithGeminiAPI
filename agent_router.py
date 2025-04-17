import traceback
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
from google import generativeai as genai
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from text_agent import TextGenerationService

class AgentRouter:
    def __init__(self, text_generation_service: TextGenerationService):
        self.text_generation_service = text_generation_service

    def route_to_agent(self, message: str, chat_history: List[BaseMessage], context: Optional[str] = None) -> str:
        """Route the message to appropriate agent and return response"""
        try:
            # Combine context with the message if available
            if context:
                augmented_message = (
                    "Using the following context to answer the question:\n\n"
                    f"{context}\n\n"
                    f"Question: {message}\n"
                    "Please provide a relevant answer based on the context above."
                )
            else:
                augmented_message = message

            # Use the augmented message with the text generation service
            response = self.text_generation_service.generate_text(
                user_message=augmented_message,
                chat_history=chat_history
            )
            return response

        except Exception as e:
            print(f"Error in agent router: {str(e)}")
            raise


