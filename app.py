from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import jwt, datetime
from functools import wraps
import os, re, fitz, json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask_mail import Mail, Message
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Configure email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '' #Add your email;
app.config['MAIL_PASSWORD'] = '' #Add your 16 word password
app.config['MAIL_DEFAULT_SENDER'] = ''
mail = Mail(app)
# CORS(app)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}}, expose_headers=["Authorization"])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 #32MB allowed
app.config['MONGO_URI'] = 'mongodb://localhost:27017/V0'
app.config['SECRET_KEY'] = 'yoursecretkey'  
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
ALLOWED_EXTENSIONS = {'pdf'}

# ---------------- JWT DECORATOR ----------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]
            print("\n\nAuthorization Header:", request.headers.get("Authorization"))
            print("\n\nToken :", token)
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.Autograder.find_one({'email': data['email']})
        except:
            return jsonify({'message': 'Invalid or expired token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if mongo.db.Autograder.find_one({'email': data['email']}):
        return jsonify({'message': 'Email already exists'}), 409
    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    verification_token = jwt.encode({
        'email': data['email'],
        'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    mongo.db.Autograder.insert_one({
        'email': data['email'],
        'password': hashed_pw,
        'verified': False,
        'marks_history': [] })
    # Send token in email body
    msg = Message('Verify your account', recipients=[data['email']])
    msg.body = f"""Hi,\n\nPlease verify your account by entering this token:\n\n{verification_token}\n\non the verification page: http://localhost:5001/verify-manual
    """
    mail.send(msg)

    return jsonify({'message': 'Registered. Check your email and verify using the token.'}), 201

@app.route('/verify/<token>')
def verify_email(token):
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        user = mongo.db.Autograder.find_one({'email': data['email']})
        if not user:
            return "User not found", 404
        if user.get('verified'):
            return "Already verified!"
        
        mongo.db.Autograder.update_one({'email': data['email']}, {'$set': {'verified': True}})
        return redirect(url_for('login_page'))
    except jwt.ExpiredSignatureError:
        return "Verification link expired", 400
    except jwt.InvalidTokenError:
        return "Invalid verification token", 400


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = mongo.db.Autograder.find_one({'email': data['email']})
    
    if not user or not bcrypt.check_password_hash(user['password'], data['password']):
        return jsonify({'message': 'Invalid credentials'}), 401

    if not user.get('verified', False):
        return jsonify({'message': 'Please verify your email before logging in.'}), 403

    token = jwt.encode({
        'email': user['email'],
        'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=2)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'token': token})


# ---------------- FILE CHECK + NLP ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_answers_from_pdf(pdf_path, is_teacher=False):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
    except Exception as e:
        return [], [], []

    answer_pattern = re.compile(r'Answer\s*\d+[:\s]', re.IGNORECASE)
    sections = answer_pattern.split(text)[1:]
    answers, keywords_list, marks = [], [], []

    for section in sections:
        if is_teacher:
            keywords_start = section.find("Keywords:")
            marks_start = section.find("Marks:")
            if marks_start == -1:
                continue
            answer_text = section[:min(keywords_start if keywords_start!=-1 else marks_start, marks_start)].strip()
            keywords = [kw.strip() for kw in section[keywords_start+9:marks_start].split(',')] if keywords_start!=-1 else []
            try:
                mark = float(section[marks_start+6:].split()[0])
            except: continue
            answers.append(answer_text)
            keywords_list.append(keywords)
            marks.append(mark)
        else:
            end = min([p for p in [section.find("Keywords:"), section.find("Marks:")] if p!=-1] or [len(section)])
            answers.append(section[:end].strip())
    return (answers, keywords_list, marks) if is_teacher else (answers, [], [])

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return set(tokens)

def calculate_keyword_matches(teacher_keywords, student_answer):
    return len(preprocess_text(" ".join(teacher_keywords)).intersection(preprocess_text(student_answer))), len(teacher_keywords)

def calculate_combined_score(teacher_answer, teacher_keywords, student_answer, marks):
    match_count, keyword_count = calculate_keyword_matches(teacher_keywords, student_answer)
    keyword_score = match_count / keyword_count if keyword_count > 0 else 0
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, teacher_answer, student_answer).ratio()
    return round(((keyword_score + similarity) / 2) * marks, 2)

def calculate_marks(teacher_answers, student_answers, teacher_keywords, marks):
    total, scores = 0, []
    for i in range(len(teacher_answers)):
        score = calculate_combined_score(teacher_answers[i], teacher_keywords[i], student_answers[i], marks[i])
        scores.append(score)
        total += score
    percent = round((total / sum(marks)) * 100, 2) if sum(marks) > 0 else 0
    return round(total, 2), percent, scores

# ---------------- COMPARE API ----------------
@app.route('/compare', methods=['POST'])
@token_required
def compare(current_user):
    teacher_file = request.files.get('teachersMarksheet')
    student_file = request.files.get('studentMarksheet')
    if not teacher_file or not student_file or not allowed_file(teacher_file.filename) or not allowed_file(student_file.filename):
        return jsonify({'error': 'Invalid files'}), 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    teacher_path = os.path.join(app.config['UPLOAD_FOLDER'], teacher_file.filename)
    student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_file.filename)
    teacher_file.save(teacher_path)
    student_file.save(student_path)

    teacher_answers, teacher_keywords, marks = extract_answers_from_pdf(teacher_path, is_teacher=True)
    student_answers, _, _ = extract_answers_from_pdf(student_path)
    student_marks, percentage, marks_list = calculate_marks(teacher_answers, student_answers, teacher_keywords, marks)

    result = {
        'student_marks': student_marks,
        'student_percentage': percentage,
        'marks_obtained_list': marks_list
    }

    mongo.db.Autograder.update_one(
        {'email': current_user['email']},
        {'$push': {'marks_history': result}}
    )
    print(f"Returning JSON: {result}") 
    return jsonify(result)

# ---------------- HTML ROUTES ----------------
@app.route('/verify-manual', methods=['GET'])
def verify_manual_page():
    return render_template('verification.html')

@app.route('/')
def home_redirect():
    return redirect(url_for('login_page'))  # Always go to login on fresh visit

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/results')
def results():
    student_marks = request.args.get('student_marks', 0)
    student_percentage = request.args.get('student_percentage', 0)
    marks_obtained_list = request.args.get('marks_obtained_list', '[]')
    return render_template("results.html", student_marks=student_marks, student_percentage=student_percentage, marks_obtained_list=json.loads(marks_obtained_list))

@app.route('/')
def root():
    if 'email' in session:
        return render_template('index.html', email=session['email'])
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)