import os

os.environ.setdefault('APP_ENV', 'test')

from app.config import get_settings

get_settings.cache_clear()
