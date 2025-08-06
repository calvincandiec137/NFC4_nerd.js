from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from modules.run_splitter import main
from modules.response import res_main



app = Flask(__name__, template_folder='../templates')
CORS(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_split():
    result = main()  # should return a dict or string
    return jsonify({"status": "success", "data": result})

@app.route('/response', methods=['POST'])
def response_gen():
    data = request.get_json()
    user_input = data.get('message', '')
    output = res_main(user_input)
    return jsonify({"status": "success", "response": output})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files part"}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append({
                "name": filename,
                "size": os.path.getsize(filepath),
                "path": filepath
            })
    
    if not uploaded_files:
        return jsonify({"status": "error", "message": "No valid files uploaded"}), 400
    
    return jsonify({
        "status": "success",
        "files": uploaded_files,
        "message": f"Successfully uploaded {len(uploaded_files)} files"
    })

@app.route('/process', methods=['POST'])
def process_files():
    try:
        # Here you would call your processing functions
        result = main()  # Replace with actual processing logic
        return jsonify({
            "status": "success",
            "message": "Files processed successfully",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)