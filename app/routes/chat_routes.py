from flask import Blueprint, request, jsonify
from flask import current_app

bp = Blueprint('chat', __name__)

@bp.route('/start_conversation', methods=['POST'])
def start_conversation():
    conversation_manager = current_app.conversation_manager
    data = request.json
    session_id = data.get('conversation_id')
    session_id = conversation_manager.start_conversation(session_id)
    return jsonify({"session_id": session_id}), 200

@bp.route('/api/chat/hr', methods=['POST'])
def send_message():
    conversation_manager = current_app.conversation_manager
    data = request.json
    session_id = data.get('conversation_id')
    message = data.get('message')
    
    if not session_id or not message:
        return jsonify({"error": "Missing session_id or message", "cost_detail_list":[]}), 400
    
    try:
        response, cost_details = conversation_manager.process_message(session_id, message)
        print(response, cost_details)
        return jsonify({
            "reply": response,
            "cost_detail_list": cost_details
        }), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e),"cost_detail_list":[]}), 400

@bp.route('/end_conversation', methods=['POST'])
def end_conversation():
    conversation_manager = current_app.conversation_manager
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    conversation_manager.end_conversation(session_id)
    return jsonify({"message": "Conversation ended successfully"}), 200