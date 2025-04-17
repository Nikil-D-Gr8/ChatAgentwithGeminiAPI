from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage
from text_agent import TextGenerationService

class AgentRouter:
    def __init__(self, text_generation_service: TextGenerationService):
        self.text_generation_service = text_generation_service

    def route_to_agent(self, message: str, chat_history: List[BaseMessage], context: Optional[str] = None) -> Dict[str, str]:
        """Route the message to appropriate agent and return response and context"""
        try:
            # Return both the context and the AI response
            return {
                "context": context if context else "",
                "message": message
            }

        except Exception as e:
            print(f"Error in agent router: {str(e)}")
            raise




