from flask import Flask

def create_app():
    app = Flask(__name__)

    # 🔐 Cần thiết để sử dụng session và flash messages
    app.secret_key = "thanhdeptrai"

    # 📁 Cấu hình thư mục upload nếu bạn cần
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # 📦 Đăng ký Blueprint
    from .routes import main
    app.register_blueprint(main)

    return app
