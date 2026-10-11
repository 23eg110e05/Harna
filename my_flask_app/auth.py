import time
import secrets
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from .models import User
from . import db

auth = Blueprint('auth', __name__)

@auth.route('/signup')
def signup():
    return render_template('signup.html')

@auth.route('/signup', methods=['POST'])
def signup_post():
    name = (request.form.get('name') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    phone = (request.form.get('phone') or '').strip()
    gender = (request.form.get('gender') or '').strip()

    # Input validations
    if not name or len(name) < 2:
        flash('Please enter a valid full name (at least 2 characters).', 'danger')
        return redirect(url_for('auth.signup'))

    if not email or '@' not in email or '.' not in email:
        flash('Please enter a valid email address.', 'danger')
        return redirect(url_for('auth.signup'))

    if len(password) < 6:
        flash('Password must be at least 6 characters long.', 'danger')
        return redirect(url_for('auth.signup'))

    # Phone validation
    parsed_phone = None
    if phone:
        if phone.isdigit() and len(phone) == 10:
            parsed_phone = int(phone)
        else:
            flash('Phone number must be a valid 10-digit number.', 'danger')
            return redirect(url_for('auth.signup'))

    gender_char = None
    if gender in ('Male', 'M'):
        gender_char = 'M'
    elif gender in ('Female', 'F'):
        gender_char = 'F'

    # Check for duplicate user
    user = User.query.filter_by(email=email).first()
    if user:
        flash('An account with this email already exists. Please log in.', 'warning')
        return redirect(url_for('auth.login'))

    new_user = User(
        name=name,
        email=email,
        password=generate_password_hash(password, method='pbkdf2:sha256'),
        phone=parsed_phone,
        gender=gender_char
    )

    db.session.add(new_user)
    try:
        db.session.commit()
        flash('Registration successful! Please sign in.', 'success')
    except Exception as e:
        print("[Harna Auth Error]", e)
        db.session.rollback()
        flash('An internal database error occurred while creating your account. Please try again.', 'danger')
        return redirect(url_for('auth.signup'))

    return redirect(url_for('auth.login'))


@auth.route('/login')
def login():
    return render_template('login.html')

@auth.route('/login', methods=['POST'])
def login_post():
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''

    if not email or not password:
        flash('Please provide both email and password.', 'error')
        return redirect(url_for('auth.login'))

    user = User.query.filter_by(email=email).first()

    if not user:
        flash('No account found with this email. Please sign up first.', 'error')
        return redirect(url_for('auth.signup'))

    if not check_password_hash(user.password, password):
        flash('Incorrect password. Please try again.', 'error')
        return redirect(url_for('auth.login'))

    session.clear()

    login_user(user, remember=False)
    session['session_id'] = secrets.token_hex(16)
    session['is_fresh_login'] = True
    session['login_time'] = time.time()
    session['user_id'] = user.id
    session['user_email'] = user.email

    response = redirect(url_for('main.dish_finder'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been signed out safely.", "success")

    response = redirect(url_for('auth.login'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
