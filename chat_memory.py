from collections import deque
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatMemory:
    def __init__(self, window_size=5, max_tokens_per_turn=100):
        self.window_size = window_size
        self.max_tokens_per_turn = max_tokens_per_turn
        self.conversation_buffer = deque(maxlen=window_size)
        self.total_turns = 0
        logger.info(f"Initialized ChatMemory with window size: {window_size}")

    def add_turn(self, user_input: str, bot_response: str):
        user_input = self._truncate_message(user_input)
        bot_response = self._truncate_message(bot_response)
        turn = {
            'turn_id': self.total_turns + 1,
            'user': user_input,
            'bot': bot_response,
            'timestamp': self._get_timestamp()
        }
        self.conversation_buffer.append(turn)
        self.total_turns += 1
        logger.debug(f"Added turn {self.total_turns} to memory buffer")

    def get_context_prompt(self, current_input: str) -> str:
        if not self.conversation_buffer:
            return f"Human: {current_input}\nAssistant:"
        context_parts = []
        for turn in self.conversation_buffer:
            context_parts.append(f"Human: {turn['user']}")
            context_parts.append(f"Assistant: {turn['bot']}")
        context_parts.append(f"Human: {current_input}")
        context_parts.append("Assistant:")
        return "\n".join(context_parts)

    def get_recent_context(self, num_turns: Optional[int] = None) -> List[Dict]:
        if num_turns is None:
            num_turns = len(self.conversation_buffer)
        return list(self.conversation_buffer)[-num_turns:]

    def clear_memory(self):
        self.conversation_buffer.clear()
        self.total_turns = 0
        logger.info("Cleared conversation memory")

    def get_memory_stats(self) -> Dict:
        return {
            'total_turns': self.total_turns,
            'current_buffer_size': len(self.conversation_buffer),
            'max_buffer_size': self.window_size,
            'buffer_full': len(self.conversation_buffer) == self.window_size
        }

    def _truncate_message(self, message: str) -> str:
        words = message.split()
        if len(words) > self.max_tokens_per_turn:
            return ' '.join(words[:self.max_tokens_per_turn]) + "..."
        return message

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def export_conversation(self) -> List[Dict]:
        return list(self.conversation_buffer)

    def import_conversation(self, conversation_data: List[Dict]):
        self.clear_memory()
        for turn in conversation_data[-self.window_size:]:
            if 'user' in turn and 'bot' in turn:
                self.conversation_buffer.append(turn)
                self.total_turns += 1
        logger.info(f"Imported {len(self.conversation_buffer)} conversation turns")

class ContextManager:
    def __init__(self, memory: ChatMemory):
        self.memory = memory
        self.current_topic = None
        self.topic_keywords = set()

    def update_topic_context(self, user_input: str, bot_response: str):
        keywords = self._extract_keywords(user_input + " " + bot_response)
        self.topic_keywords.update(keywords)
        if len(self.topic_keywords) > 20:
            self.topic_keywords = set(list(self.topic_keywords)[-15:])

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                      'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words = text.lower().split()
        keywords = [word.strip('.,!?;:"()[]') for word in words
                    if len(word) > 3 and word not in stop_words]
        return keywords[:5]

    def get_topic_context(self) -> str:
        if self.topic_keywords:
            return f"Current conversation topics: {', '.join(list(self.topic_keywords)[-5:])}"
        return "No specific topic context available."
