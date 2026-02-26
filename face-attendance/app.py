from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, send_file
import os
import csv
from datetime import datetime
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pandas as pd
from models import db, Admin, User, Attendance

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'tmp_uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# --- FACE RECOGNITION LOGIC ---
def recognize_face_in_image(image):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    names = {}
    if os.path.exists("trainer.yml") and os.path.exists("names.pickle"):
        recognizer.read("trainer.yml")
        with open("names.pickle", "rb") as f:
            names = pickle.load(f)
    else:
        return "System not trained", 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(detected_faces) == 0:
        detected_faces = profile_cascade.detectMultiScale(gray, 1.3, 5)

    if len(detected_faces) > 0:
        (x, y, bw, bh) = detected_faces[0]
        if names and bw > 0 and bh > 0:
            face_roi = gray[y:y+bh, x:x+bw]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)
            id_label, confidence = recognizer.predict(face_roi)
            
            if confidence < 100:
                name = names.get(id_label, "Unknown")
                return name, confidence
                
    return "Unknown", 100

def db_mark_attendance(name):
    user = User.query.filter_by(name=name).first()
    if user:
        now = datetime.now()
        att = Attendance(
            user_id=user.id,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            status='Present'
        )
        db.session.add(att)
        db.session.commit()
        return True
    return False

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.password == password: # Simple for this demo
            login_user(admin)
            return redirect(url_for('admin_dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        files = request.files.getlist('photos')
        
        if username and files:
            # Update DB
            existing_user = User.query.filter_by(name=username).first()
            if not existing_user:
                new_user = User(name=username)
                db.session.add(new_user)
                db.session.commit()
            
            user_dir = f"dataset/faces/{username}"
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            count = 0
            for file in files:
                if file.filename == '': continue
                filename = secure_filename(file.filename)
                filepath = os.path.join(user_dir, f"{username}_{count}.jpg")
                file.save(filepath)
                count += 1
            
            from train_model import train_model
            trained = train_model()
            
            if trained:
                flash(f'Successfully registered {username}', 'success')
                return render_template('register.html', success=True, username=username, count=count)
            else:
                flash('Training failed. No faces detected in uploaded images.', 'error')
                return render_template('register.html', error="Failed to detect face.", username=username)
            
    return render_template('register.html')

@app.route('/attendance_upload', methods=['POST'])
def attendance_upload():
    if 'photo' not in request.files: return redirect(url_for('index'))
    file = request.files['photo']
    if file.filename == '': return redirect(url_for('index'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        os.remove(filepath)
        return render_template('attendance_result.html', name="Invalid File", status="Failed", confidence=0)
        
    name, confidence = recognize_face_in_image(image)
    if name != "Unknown" and name != "System not trained":
        db_mark_attendance(name)
        status = "Success"
    else:
        status = "Failed"
    
    os.remove(filepath)
    return render_template('attendance_result.html', name=name, status=status, confidence=round(100-confidence))

@app.route('/attendance')
def view_attendance():
    all_attendance = Attendance.query.order_by(Attendance.id.desc()).all()
    return render_template('attendance.html', attendance=all_attendance)

@app.route('/admin')
@login_required
def admin_dashboard():
    users = User.query.all()
    user_count = User.query.count()
    attendance_count = Attendance.query.count()
    return render_template('admin_dashboard.html', users=users, user_count=user_count, attendance_count=attendance_count)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    # Remove images
    user_dir = f"dataset/faces/{user.name}"
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
    
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.name} and all data removed.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/export')
@login_required
def export_attendance():
    records = Attendance.query.all()
    data = []
    for r in records:
        data.append({
            'ID': r.id,
            'Name': r.user.name,
            'Date': r.date,
            'Time': r.time,
            'Status': r.status
        })
    df = pd.DataFrame(data)
    export_path = 'attendance_report.xlsx'
    df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        if not Admin.query.filter_by(username='admin').first():
            admin = Admin(username='admin', password='admin123')
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True, port=5000, use_reloader=False)
