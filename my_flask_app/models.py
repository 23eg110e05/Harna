from . import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(80),nullable=False) #make all these fields required in html
    email = db.Column(db.String(100),unique=True,nullable=False)
    password = db.Column(db.String(150),nullable=False)
    phone=db.Column(db.Integer)
    gender=db.Column(db.String(1))
    favourites = db.relationship('Favourite',backref='author',lazy=True)

    def set_password(self, password):
        """Set the password to a hashed value."""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Check if the provided password matches the stored hashed password."""
        return check_password_hash(self.password, password)

class Favourite(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    user_id=db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    recipe_id=db.Column(db.Integer,nullable=False)