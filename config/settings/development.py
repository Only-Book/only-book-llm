from .base import *

DEBUG = True

ALLOWED_HOSTS = []

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": 'onlybook',
        "USER": 'admin',
        "PASSWORD": '60221210',
        "HOST": 'database-1.cby0ykimabta.ap-northeast-2.rds.amazonaws.com',
        "PORT": '3306',
    }
}

STATICFILES_DIRS = [
    BASE_DIR / 'static'
]