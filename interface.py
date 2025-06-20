import sys
import signal
import time
from typing import Optional
import logging

from model_loader import ModelLoader, get_recommended_model
from chat_memory import ChatMemory, ContextManager

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self, model_name: Optional[str] = None, memory_window: int = 5):
        # Initialize model, memory, and context manager
        self.model_name = model_name or get_recommended_model()
        self.model_loader = ModelLoader(self.model_name)
        self.memory = ChatMemory(window_size=memory_window)
        self.context_manager = ContextManager(self.memory)
        self.running = False
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info(f"Initialized ChatInterface with model: {self.model_name}")
    
    def initialize(self) -> bool:
        # Load the model
        print("🤖 Initializing chatbot...")
        print(f"📦 Loading model: {self.model_name}")
        
        try:
            success = self.model_loader.load_model()
            if success:
                print("✅ Model loaded successfully!")
                print(f"🧠 Memory window size: {self.memory.window_size} turns")
                print(f"💻 Device: {self.model_loader.device}")
                return True
            else:
                print("❌ Failed to load model!")
                return False
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    def start_chat(self):
        # Start the main chat loop
        if not self.initialize():
            return
        
        self.running = True
        self._print_welcome_message()
        
        try:
            while self.running:
                user_input = self._get_user_input()
                if user_input is None:
                    break
                if self._handle_special_commands(user_input):
                    continue
                response = self._generate_response(user_input)
                self._display_response(response)
                self.memory.add_turn(user_input, response)
                self.context_manager.update_topic_context(user_input, response)
        except KeyboardInterrupt:
            self._handle_exit()
        except Exception as e:
            logger.error(f"Chat loop error: {str(e)}")
            print(f"\n❌ An error occurred: {str(e)}")
        finally:
            self._cleanup()
    
    def _print_welcome_message(self):
        print("\n" + "="*50)
        print("🤖 Welcome to the Local AI Chatbot!")
        print("="*50)
        print("Type your messages and press Enter to chat.")
        print("Commands:")
        print("  /exit    - Exit the chatbot")
        print("  /clear   - Clear conversation memory")
        print("  /stats   - Show memory statistics")
        print("  /help    - Show this help message")
        print("-"*50)
    
    def _get_user_input(self) -> Optional[str]:
        # Get input from user
        try:
            return input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return None
    
    def _handle_special_commands(self, user_input: str) -> bool:
        # Handle commands like /exit, /clear, etc.
        if not user_input.startswith('/'):
            return False
        
        command = user_input.lower().strip()
        if command == '/exit':
            self._handle_exit()
        elif command == '/clear':
            self.memory.clear_memory()
            print("🧹 Conversation memory cleared!")
        elif command == '/stats':
            self._show_stats()
        elif command == '/help':
            self._show_help()
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help to see available commands.")
        return True
    
    def _generate_response(self, user_input: str) -> str:
        # Generate response using model
        try:
            print("🤔 Thinking...", end="", flush=True)
            prompt = self.memory.get_context_prompt(user_input)
            response = self.model_loader.generate_response(prompt)
            print("\r" + " " * 15 + "\r", end="")
            return response
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _display_response(self, response: str):
        # Show model response
        print(f"🤖 Bot: {response}")
    
    def _show_stats(self):
        # Display memory and model stats
        stats = self.memory.get_memory_stats()
        model_info = self.model_loader.get_model_info()
        print("\n📊 Chatbot Statistics:")
        print(f"  • Total conversation turns: {stats['total_turns']}")
        print(f"  • Current memory buffer: {stats['current_buffer_size']}/{stats['max_buffer_size']}")
        print(f"  • Memory buffer full: {'Yes' if stats['buffer_full'] else 'No'}")
        print(f"  • Model: {model_info['model_name']}")
        print(f"  • Device: {model_info['device']}")
        recent_turns = self.memory.get_recent_context(3)
        if recent_turns:
            print(f"  • Recent topics: {self.context_manager.get_topic_context()}")
    
    def _show_help(self):
        # Show help message
        print("\n❓ Available Commands:")
        print("  /exit    - Exit the chatbot gracefully")
        print("  /clear   - Clear the conversation memory")
        print("  /stats   - Show memory and model statistics")
        print("  /help    - Show this help message")
    
    def _handle_exit(self):
        # Exit gracefully
        print("\n👋 Exiting chatbot. Goodbye!")
        stats = self.memory.get_memory_stats()
        if stats['total_turns'] > 0:
            print(f"📈 Total conversation turns: {stats['total_turns']}")
        self.running = False
    
    def _signal_handler(self, signum, frame):
        print("\n\n⚠️  Received interrupt signal...")
        self._handle_exit()
    
    def _cleanup(self):
        logger.info("Performing cleanup...")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Local AI Chatbot')
    parser.add_argument('--model', '-m', type=str, help='Hugging Face model name to use')
    parser.add_argument('--memory', '-w', type=int, default=5, help='Conversation memory window size (default: 5)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        chatbot = ChatInterface(model_name=args.model, memory_window=args.memory)
        chatbot.start_chat()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
