from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)   # None for Google users
    google_id     = db.Column(db.String(255), unique=True, nullable=True)
    display_name  = db.Column(db.String(255), nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    baskets       = db.relationship('Basket', backref='user', lazy=True,
                                    cascade='all, delete-orphan')

class Basket(db.Model):
    __tablename__ = 'baskets'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name        = db.Column(db.String(255), nullable=True)          # optional label
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    # parameters used for generation
    target_calories = db.Column(db.Float)
    target_protein  = db.Column(db.Float, nullable=True)
    target_carbs    = db.Column(db.Float, nullable=True)
    target_fat      = db.Column(db.Float, nullable=True)
    mode            = db.Column(db.String(10))   # 'soft' | 'cap'
    # totals
    total_cost      = db.Column(db.Float)
    total_calories  = db.Column(db.Float)
    total_protein   = db.Column(db.Float)
    total_carbs     = db.Column(db.Float)
    total_fat       = db.Column(db.Float)

    items = db.relationship('BasketItem', backref='basket', lazy=True,
                            cascade='all, delete-orphan')

class BasketItem(db.Model):
    __tablename__ = 'basket_items'
    id                = db.Column(db.Integer, primary_key=True)
    basket_id         = db.Column(db.Integer, db.ForeignKey('baskets.id'), nullable=False)
    product_name      = db.Column(db.String(255))
    pkg_label         = db.Column(db.String(50))    # "1 კგ", "500 მლ" etc.
    units             = db.Column(db.Float)          # შეფუთვების რაოდენობა
    recommended_grams = db.Column(db.Float)          # ოპტიმიზერის რეკომენდაცია
    remaining_grams   = db.Column(db.Float)          # package_size*units - recommended
    cost              = db.Column(db.Float)
    calories          = db.Column(db.Float)
    protein           = db.Column(db.Float)
    carbs             = db.Column(db.Float)
    fat               = db.Column(db.Float)
