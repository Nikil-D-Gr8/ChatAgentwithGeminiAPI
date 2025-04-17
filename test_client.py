
import requests
import json
import uuid
import base64
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import io

class ChatClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session_id = None
        self._session = requests.Session()  # Use a persistent session
    
    def chat(self, message: str, image_path: Optional[str] = None):
        """Send a chat message with optional image to the server."""
        url = f'{self.base_url}/chat'
        headers = {'Content-Type': 'application/json'}
        
        # Always include session_id if we have one
        data = {
            'message': message,
            'session_id': self.session_id
        }
        
        if image_path:
            try:
                image_data = self._encode_image(image_path)
                data['image'] = image_data
            except Exception as e:
                print(f"Error processing image: {e}")
                return None
        
        try:
            print(f"Sending request with session_id: {self.session_id}")
            response = self._session.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            # Update session_id from response
            if 'session_id' in result:
                if not self.session_id:
                    print(f"New session created: {result['session_id']}")
                elif self.session_id != result['session_id']:
                    print(f"Session ID changed from {self.session_id} to {result['session_id']}")
                self.session_id = result['session_id']
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Server response: {e.response.text}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            if 'response' in locals():
                print(f"Raw response: {response.text}")
            return None

    def _encode_image(self, image_path: str) -> str:
        """Load image using PIL and convert to base64 string"""
        try:
            # Open and convert image to RGB
            with Image.open(image_path).convert('RGB') as img:
                # Save image to bytes buffer
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                # Convert to base64 string
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return img_str
        except Exception as e:
            print(f"Error encoding image: {e}")
            raise

    def reset_session(self):
        """Reset the current chat session."""
        if not self.session_id:
            print("No active session to reset")
            return
            
        url = f'{self.base_url}/reset'
        data = {'session_id': self.session_id}
        
        try:
            response = self._session.post(url, json=data)
            response.raise_for_status()
            print(f"Session {self.session_id} reset successfully")
            self.session_id = None
        except requests.exceptions.RequestException as e:
            print(f"Error resetting session: {e}")

def main():
    client = ChatClient()
    print("Chat API Client")
    print("Commands:")
    print("  'quit' - Exit the program")
    print("  'reset' - Start new session")
    print("  'image \"<path>\" <message>' - Send message with an image (path must be in quotes)")
    print(f"Session ID: {client.session_id}")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Handle commands
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            client.reset_session()
            continue
        elif user_input.lower().startswith('image '):
            # Handle image + text input
            try:
                # Look for text between quotes after "image"
                import re
                match = re.search(r'image\s*"([^"]*)"(.*)', user_input, re.IGNORECASE)
                if not match:
                    print('Please provide image path in quotes. Example: image "path/to/image.jpg" Your message here')
                    continue
                
                image_path = match.group(1).strip()  # Get the path from between quotes
                message = match.group(2).strip()  # Get the rest of the message
                
                if not message:
                    print("Please provide a message after the image path")
                    continue
                
                if not Path(image_path).exists():
                    print(f"Error: Image file not found: {image_path}")
                    continue
                    
                result = client.chat(message, image_path)
                
                # Display response
                if result:
                    if 'response' in result:
                        print(f"\nAI: {result['response']}")
                    elif 'error' in result:
                        print(f"\nError: {result['error']}")
                    else:
                        print("\nError: Unexpected response format")
                else:
                    print("\nError: No response received")
                    
            except Exception as e:
                print(f"\nError processing image request: {str(e)}")
                
        else:
            # Handle text-only input
            result = client.chat(user_input)
            
            # Display response
            if result:
                if 'response' in result:
                    print(f"\nAI: {result['response']}")
                elif 'error' in result:
                    print(f"\nError: {result['error']}")
                else:
                    print("\nError: Unexpected response format")
            else:
                print("\nError: No response received")

if __name__ == "__main__":
    main()








