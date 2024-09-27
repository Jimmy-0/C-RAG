import uuid
from app.services.llm import HrTalk
from app.utils.load_data import LoadHRdata

class ConversationManager:
    def __init__(self, load_data: LoadHRdata):
        self.sessions = {}
        self.vector_store = load_data

    def start_conversation(self, session_id):
        self.sessions[session_id] = HrTalk(self.vector_store)
        return session_id

    def process_message(self, session_id, message):
        print(self.sessions)
        if session_id not in self.sessions:
            self.start_conversation(session_id)
        
        hr_talk = self.sessions[session_id]
        print(hr_talk)
        response, cost_details = hr_talk.chat_with_follow_up({
            'message': message,
            'current_conversation_id': session_id
        })
        return response, cost_details

    def end_conversation(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _generate_unique_id(self):
        return str(uuid.uuid4())