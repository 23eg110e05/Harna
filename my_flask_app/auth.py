from flask import Blueprint,render_template,redirect,url_for,request,flash,session, current_app
from werkzeug.security import generate_password_hash,check_password_hash
from .models import User
from . import db
from my_flask_app.main import main
from flask_login import login_user,logout_user,login_required, current_user
from datetime import datetime, timedelta
import time
import secrets

auth = Blueprint('auth',__name__)

# Wtch tutorial 6 for more nice management of database

@auth.route('/signup')
def signup():
    return render_template('signup.html')

@auth.route('/signup',methods = ['POST'])
def signup_post():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    phone=request.form.get('phone')
    gender=request.form.get('gender')

    # if len(password) < 6:                   #Add more security checks later      
    #     flash('Password must be at least 6 characters long.', 'danger')
    #     return redirect(url_for('auth.signup'))

    # #debug
    # print("Signup: ",name,email,password)
    user = User.query.filter_by(email=email).first()
    if user:
        flash('User already exists. Please log in.', 'danger')
        return redirect(url_for('auth.login'))
    
    new_user = User(
        name = name,
        email = email,
        password = generate_password_hash(password,method='pbkdf2:sha256'),#Why did we choose this?
        phone=phone,
        gender=gender  
    )

    db.session.add(new_user)
    try:
        db.session.commit()
        flash('Signup Successful!','success')
    except Exception as e:
        print("Error during commit:", e)
        db.session.rollback()
        return "An error occurred while signing up.", 500
    
    return redirect(url_for('auth.login'))


@auth.route('/login')
def login():
    
    if current_user.is_authenticated:
        logout_user()
        session.clear()
    return render_template('login.html')

@auth.route('/login',methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    # Always set remember to False to disable the remember me functionality
    remember = False

    user = User.query.filter_by(email=email).first()

    if not user:
        flash('User with provided E-mail does not exist please signup','error')
        return redirect(url_for('auth.signup'))
    if not check_password_hash(user.password,password):
        flash('Password is incorrect, please try again!','error')
        return redirect(url_for('auth.login'))
    
    
    session.clear()
    
    # Set remember=False to disable the remember me functionality
    login_user(user, remember=remember)
    
    # Generate a unique session ID
    session['session_id'] = secrets.token_hex(16)
    
    # Set a flag in the session to indicate this is a fresh login
    session['is_fresh_login'] = True
    session['login_time'] = time.time()
    session['user_id'] = user.id
    session['user_email'] = user.email
    
    # Log login attempt
    print(f"User {user.email} logged in at {datetime.now()}")
    
    # Clear any cached pages
    response = redirect(url_for('main.dish_finder'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@auth.route('/logout')
@login_required
def logout():
    
    if current_user.is_authenticated:
        print(f"User {current_user.email} logged out at {datetime.now()}")
    
  
    logout_user()
    
    # Clear the session completely
    session.clear()
    
    # Show logout message
    flash("You have been successfully logged out.", "success")
    
    # Redirect with cache control headers
    response = redirect(url_for('auth.login'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
