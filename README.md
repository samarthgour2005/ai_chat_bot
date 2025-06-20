Project Structure
├── model_loader.py     # Loads and manages the Hugging Face model
├── chat_memory.py      # Handles memory and conversation history
├── interface.py        # Command-line interface and main chatbot loop
├── README.md           # This file

🚀 Features
💬 Fully functional CLI chatbot
🧠 Maintains memory of last N (default 5) conversation turns
🤖 Uses FLAN-T5 model for accurate responses
📦 Runs locally on CPU or GPU (optional)
📂 Modular and maintainable Python codebase
🛑 Graceful exit with /exit, memory control with /clear, and /stats

💡 How It Works
The chatbot loads a Hugging Face model using the text2text-generation pipeline.
It prepends prompts like Answer this question: for better model performance.
A sliding window memory keeps track of recent conversations and builds context for each response.        
Commands such as /exit, /clear, and /stats enhance user control over the session.

🧪 Sample Interaction
yaml
Copy
Edit
🤖 Welcome to the Local AI Chatbot!
    Type your messages and press Enter to chat.            

👤 You: What is the capital of France?
🤖 Bot: The capital of France is Paris.

👤 You: And Italy?
🤖 Bot: The capital of Italy is Rome.

👤 You: /stats
📊 Chatbot Statistics:
  • Total conversation turns: 2
  • Model: google/flan-t5-base
  • Device: cpu

👤 You: /exit
👋 Exiting chatbot. Goodbye!
