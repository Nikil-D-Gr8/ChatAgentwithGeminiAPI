from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage, AIMessage
from text_agent import TextGenerationService
from image_agent import PropertyIssueDetectionAgent
from RAGsystem import RAGSystem

class AgentRouter:
    def __init__(
        self, 
        text_generation_service: TextGenerationService, 
        image_agent: PropertyIssueDetectionAgent,
        rag_system: RAGSystem
    ):
        self.text_generation_service = text_generation_service
        self.image_agent = image_agent
        self.rag_system = rag_system

    def _handle_image_request(
        self, 
        message: str, 
        image_data: str,
        chat_history: List[BaseMessage]
    ) -> Dict[str, str]:
        """Handle image-based requests"""
        # Get image analysis
        image_analysis = self.image_agent.analyze_image(image_data, message)
        
        # Create context from image analysis
        image_context = f"Image Analysis:\n{image_analysis['description']}"
        if image_analysis['detected_issues']:
            issues_text = "\nDetected Issues:\n" + "\n".join(
                [f"- {issue['issue']}: {issue['description']} (Severity: {issue['severity']})" 
                 for issue in image_analysis['detected_issues']]
            )
            image_context += issues_text

        # Generate response using image analysis as context
        response = self.text_generation_service.generate_response(
            user_message=message,
            chat_history=chat_history,
            context=image_context
        )

        return {
            "response": response,
            "context": image_context
        }

    def _handle_text_request(
        self, 
        message: str, 
        chat_history: List[BaseMessage]
    ) -> Dict[str, str]:
        """Handle text-based requests"""
        # Get relevant context from RAG system
        context = self.rag_system.get_relevant_context(message)
        
        # Generate response using retrieved context
        response = self.text_generation_service.generate_response(
            user_message=message,
            chat_history=chat_history,
            context=context
        )

        return {
            "response": response,
            "context": context
        }

    def route_message(
        self, 
        message: str, 
        chat_history: List[BaseMessage], 
        image_data: Optional[str] = None
    ) -> Dict[str, str]:
        """Route the message to appropriate agent and return response"""
        try:
            # Route based on presence of image
            if image_data:
                result = self._handle_image_request(message, image_data, chat_history)
            else:
                result = self._handle_text_request(message, chat_history)

            return result

        except Exception as e:
            print(f"Error in agent router: {str(e)}")
            raise





