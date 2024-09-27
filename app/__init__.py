from flask import Flask
from app.services.conversation_manager import ConversationManager
from app.utils.load_data import LoadHRdata

def create_app():
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object('config')
    
    # Initialize services
    load_data = LoadHRdata(data_dir='data_files', db_path='hr_data.db')

    # vector_store = load_data.pdf_loader()
    load_data.pdf_loader()
    conversation_manager = ConversationManager(load_data)
    
    # Attach services to app
    app.conversation_manager = conversation_manager
    
    # Register blueprints
    from app.routes import chat_routes
    app.register_blueprint(chat_routes.bp)
    
    return app