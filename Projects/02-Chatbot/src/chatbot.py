"""
Main Chatbot Implementation

This module provides the main chatbot class that integrates all components:
intent classification, entity extraction, dialogue management, and response generation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from intent_classifier import IntentClassifier
from entity_extractor import EntityExtractor
from dialogue_manager import DialogueManager
from response_generator import ResponseGenerator


@dataclass
class ChatbotResponse:
    """Structured response from the chatbot"""
    text: str
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    context: Dict[str, Any]
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    user_input: str
    bot_response: ChatbotResponse
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationSession:
    """Manages conversation state for a single user session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: List[ConversationTurn] = []
        self.context: Dict[str, Any] = {}
        self.user_profile: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_turn(self, user_input: str, bot_response: ChatbotResponse):
        """Add a conversation turn"""
        turn = ConversationTurn(user_input, bot_response)
        self.turns.append(turn)
        self.last_activity = datetime.now()
    
    def get_recent_context(self, num_turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation turns for context"""
        return self.turns[-num_turns:] if self.turns else []
    
    def update_context(self, key: str, value: Any):
        """Update conversation context"""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from conversation context"""
        return self.context.get(key, default)


class Chatbot:
    """
    Main chatbot class that orchestrates all NLU and dialogue components
    """
    
    def __init__(self, config_path: str = "config/chatbot_config.json"):
        """
        Initialize the chatbot with all components
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.intent_classifier = IntentClassifier(
            model_path=self.config.get("intent_model_path", "models/intent_model.pkl")
        )
        
        self.entity_extractor = EntityExtractor(
            model_path=self.config.get("entity_model_path", "models/entity_model.pkl")
        )
        
        self.dialogue_manager = DialogueManager(
            config=self.config.get("dialogue_config", {})
        )
        
        self.response_generator = ResponseGenerator(
            templates_path=self.config.get("templates_path", "data/response_templates.json"),
            model_path=self.config.get("response_model_path", "models/response_model.pkl")
        )
        
        # Session management
        self.sessions: Dict[str, ConversationSession] = {}
        self.default_session_timeout = self.config.get("session_timeout", 3600)  # 1 hour
        
        self.logger.info("Chatbot initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "intent_threshold": 0.7,
            "entity_threshold": 0.6,
            "max_conversation_turns": 50,
            "session_timeout": 3600,
            "fallback_responses": [
                "I'm not sure I understand. Could you please rephrase that?",
                "I didn't quite get that. Can you try asking in a different way?",
                "I'm still learning. Could you help me understand what you mean?"
            ],
            "greeting_responses": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Welcome! How may I assist you?"
            ]
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("chatbot")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        session_id = user_id or str(uuid.uuid4())
        self.sessions[session_id] = ConversationSession(session_id)
        self.logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id)
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        time_since_activity = (datetime.now() - session.last_activity).seconds
        if time_since_activity > self.default_session_timeout:
            self.logger.info(f"Session expired: {session_id}")
            self.sessions[session_id] = ConversationSession(session_id)
        
        return self.sessions[session_id]
    
    def respond(self, user_input: str, session_id: str = "default") -> ChatbotResponse:
        """
        Generate response to user input
        
        Args:
            user_input: User's message
            session_id: Session identifier
        
        Returns:
            Structured chatbot response
        """
        try:
            # Get or create session
            session = self.get_session(session_id)
            
            # Preprocess input
            processed_input = self._preprocess_input(user_input)
            
            # Intent classification
            intent, intent_confidence = self.intent_classifier.predict(processed_input)
            
            # Entity extraction
            entities = self.entity_extractor.extract(processed_input)
            
            # Update dialogue state
            dialogue_state = self.dialogue_manager.update_state(
                user_input=processed_input,
                intent=intent,
                entities=entities,
                context=session.context,
                conversation_history=session.get_recent_context()
            )
            
            # Generate response
            response_text = self.response_generator.generate(
                intent=intent,
                entities=entities,
                dialogue_state=dialogue_state,
                context=session.context
            )
            
            # Create structured response
            bot_response = ChatbotResponse(
                text=response_text,
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                context=session.context.copy(),
                session_id=session_id,
                metadata={
                    "dialogue_state": dialogue_state,
                    "processing_time": 0.0  # Would be calculated in real implementation
                }
            )
            
            # Update session
            session.add_turn(user_input, bot_response)
            session.context.update(dialogue_state.get("context_updates", {}))
            
            self.logger.info(f"Generated response for session {session_id}: {intent}")
            return bot_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return self._create_error_response(session_id, str(e))
    
    def _preprocess_input(self, user_input: str) -> str:
        """Preprocess user input"""
        # Basic preprocessing
        processed = user_input.strip()
        
        # Handle empty input
        if not processed:
            return "hello"  # Default to greeting
        
        return processed
    
    def _create_error_response(self, session_id: str, error_message: str) -> ChatbotResponse:
        """Create error response"""
        fallback_responses = self.config.get("fallback_responses", [
            "I'm sorry, I encountered an error. Please try again."
        ])
        
        return ChatbotResponse(
            text=fallback_responses[0],
            intent="error",
            confidence=0.0,
            entities=[],
            context={},
            session_id=session_id,
            metadata={"error": error_message}
        )
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        history = []
        for turn in session.turns:
            history.append({
                "timestamp": turn.timestamp.isoformat(),
                "user_input": turn.user_input,
                "bot_response": turn.bot_response.text,
                "intent": turn.bot_response.intent,
                "confidence": turn.bot_response.confidence,
                "entities": turn.bot_response.entities
            })
        
        return history
    
    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleared session: {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        intents = [turn.bot_response.intent for turn in session.turns]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "session_id": session_id,
            "total_turns": len(session.turns),
            "session_duration": (session.last_activity - session.created_at).seconds,
            "intent_distribution": intent_counts,
            "average_confidence": sum(turn.bot_response.confidence for turn in session.turns) / len(session.turns) if session.turns else 0,
            "context_size": len(session.context)
        }
    
    def batch_respond(self, inputs: List[Tuple[str, str]]) -> List[ChatbotResponse]:
        """
        Process multiple inputs in batch
        
        Args:
            inputs: List of (user_input, session_id) tuples
        
        Returns:
            List of chatbot responses
        """
        responses = []
        for user_input, session_id in inputs:
            response = self.respond(user_input, session_id)
            responses.append(response)
        
        return responses
    
    def export_conversations(self, session_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export conversation data for analysis"""
        if session_ids is None:
            session_ids = list(self.sessions.keys())
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "sessions": {}
        }
        
        for session_id in session_ids:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                export_data["sessions"][session_id] = {
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "turns": [
                        {
                            "timestamp": turn.timestamp.isoformat(),
                            "user_input": turn.user_input,
                            "bot_response": {
                                "text": turn.bot_response.text,
                                "intent": turn.bot_response.intent,
                                "confidence": turn.bot_response.confidence,
                                "entities": turn.bot_response.entities
                            }
                        }
                        for turn in session.turns
                    ],
                    "context": session.context,
                    "stats": self.get_session_stats(session_id)
                }
        
        return export_data


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Chatbot")
    parser.add_argument("--config", default="config/chatbot_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--test", action="store_true", 
                       help="Run test conversations")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = Chatbot(args.config)
    
    if args.interactive:
        print("🤖 Chatbot initialized! Type 'quit' to exit.")
        session_id = chatbot.create_session()
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = chatbot.get_session_stats(session_id)
                print(f"📊 Session Stats: {json.dumps(stats, indent=2)}")
                continue
            
            if user_input.lower() == 'history':
                history = chatbot.get_conversation_history(session_id)
                print(f"📜 Conversation History: {json.dumps(history, indent=2)}")
                continue
            
            response = chatbot.respond(user_input, session_id)
            print(f"🤖 Bot: {response.text}")
            print(f"   Intent: {response.intent} (confidence: {response.confidence:.2f})")
            if response.entities:
                print(f"   Entities: {response.entities}")
    
    elif args.test:
        print("🧪 Running test conversations...")
        
        test_inputs = [
            ("Hello", "test_session_1"),
            ("I want to book a flight", "test_session_1"),
            ("To Paris on Monday", "test_session_1"),
            ("What's the weather like?", "test_session_2"),
            ("Thank you", "test_session_1")
        ]
        
        responses = chatbot.batch_respond(test_inputs)
        
        for i, (user_input, session_id) in enumerate(test_inputs):
            response = responses[i]
            print(f"\nTest {i+1}:")
            print(f"  Input: {user_input}")
            print(f"  Response: {response.text}")
            print(f"  Intent: {response.intent} ({response.confidence:.2f})")
            print(f"  Session: {session_id}")
        
        # Print session statistics
        for session_id in ["test_session_1", "test_session_2"]:
            stats = chatbot.get_session_stats(session_id)
            if stats:
                print(f"\n📊 Stats for {session_id}:")
                print(f"  Total turns: {stats['total_turns']}")
                print(f"  Intent distribution: {stats['intent_distribution']}")
    
    else:
        print("Use --interactive for chat mode or --test for testing")
        print("Example: python chatbot.py --interactive")