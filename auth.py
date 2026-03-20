from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.app_page'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        name     = request.form.get('name', '').strip()

        if not email or not password:
            flash('email და პაროლი სავალდებულოა', 'error')
            return redirect(url_for('auth.register'))
        if len(password) < 8:
            flash('პაროლი მინიმუმ 8 სიმბოლო უნდა იყოს', 'error')
            return redirect(url_for('auth.register'))
        if User.query.filter_by(email=email).first():
            flash('ეს email უკვე რეგისტრირებულია', 'error')
            return redirect(url_for('auth.register'))

        user = User(
            email         = email,
            password_hash = generate_password_hash(password),
            display_name  = name or email.split('@')[0]
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('main.app_page'))

    return render_template('register.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.app_page'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()

        if not user or not user.password_hash:
            flash('email ან პაროლი არასწორია', 'error')
            return redirect(url_for('auth.login'))
        if not check_password_hash(user.password_hash, password):
            flash('email ან პაროლი არასწორია', 'error')
            return redirect(url_for('auth.login'))

        login_user(user)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('main.app_page'))

    return render_template('login.html')


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
