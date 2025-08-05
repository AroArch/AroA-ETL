import secrets
from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import os

backend = default_backend()
iterations = 100_000

def _derive_key(password: bytes, salt: bytes, iterations: int = iterations) -> bytes:
    """Derive a secret key from a given password and salt"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt,
        iterations=iterations, backend=backend)
    return b64e(kdf.derive(password))

def password_encrypt(message: bytes, password: str, iterations: int = iterations) -> bytes:
    salt = secrets.token_bytes(16)
    key = _derive_key(password.encode(), salt, iterations)
    return b64e(
        b'%b%b%b' % (
            salt,
            iterations.to_bytes(4, 'big'),
            b64d(Fernet(key).encrypt(message)),
        )
    )

def password_decrypt(token: bytes, password: str) -> bytes:
    decoded = b64d(token)
    salt, iter, token = decoded[:16], decoded[16:20], b64e(decoded[20:])
    iterations = int.from_bytes(iter, 'big')
    key = _derive_key(password.encode(), salt, iterations)
    return Fernet(key).decrypt(token)

def create_credentials_file_interactive(password: str, fname: str) -> None:
    print("Enter user name:")
    dbuser = input()
    print("Enter db password:")
    dbpass = input()
    print("Enter db ip address:")
    dbaddress = input()
    print("Enter db name:")
    dbname = input()
    assert "," not in dbpass, "',' found in password. Use different separator symbol for encryption message"
    message = f"{dbuser},{dbpass},{dbaddress},{dbname}"
    save_credentials_to_file(message, password, fname)

def read_credentials_from_file(password: str, fname: str):
    if not os.path.isfile(fname):
        create_credentials_file_interactive(password, fname)
    with open(fname,"rb") as cfile:
        dbuser,dbpass,dbaddress,dbname = password_decrypt(cfile.readline(),password).decode().split(",")
    return dbuser, dbpass, dbaddress, dbname 

def save_credentials_to_file(message:str, password:str, fname:str):
    enc_message = password_encrypt(message.encode(), password)
    with open(fname,"wb") as cfile:
        cfile.write(enc_message)
