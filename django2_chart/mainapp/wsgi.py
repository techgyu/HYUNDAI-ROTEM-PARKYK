"""
WSGI config for mainapp project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os  # OS 모듈 임포트

from django.core.wsgi import get_wsgi_application  # Django의 WSGI 애플리케이션 함수 임포트

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mainapp.settings")  # 환경변수에 settings 모듈 지정

application = get_wsgi_application()  # WSGI 애플리케이션 객체
