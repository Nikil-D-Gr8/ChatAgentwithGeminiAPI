import traceback
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
from google import generativeai as genai
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from text_agent import TextGenerationService

class AgentRouter:
    def __init__(self, text_generation_service: TextGenerationService):
        self.text_gen_service = text_generation_service

    def route_to_agent(self, message: str, chat_history: List[BaseMessage]) -> str:
        """Route the message to appropriate agent and return response"""
        print(f"\nRouting message with chat history length: {len(chat_history)}")
        
        try:
            # Format chat history for the LLM
            formatted_history = []
            for msg in chat_history:
                if isinstance(msg, SystemMessage):
                    formatted_history.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    formatted_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_history.append({"role": "assistant", "content": msg.content})
            
            print("Formatted chat history:")
            for idx, msg in enumerate(formatted_history):
                print(f"{idx}: {msg['role']} - {msg['content'][:50]}...")

            # Generate response using the text generation service
            response = self.text_gen_service.generate_response(
                message=message,
                chat_history=formatted_history
            )
            
            return response

        except Exception as e:
            print(f"Error in AgentRouter: {str(e)}")
            traceback.print_exc()
            return f"I apologize, but I encountered an error: {str(e)}"

