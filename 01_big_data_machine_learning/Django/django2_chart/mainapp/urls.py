"""
URL configuration for mainapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin  # 관리자 사이트 모듈 임포트
from django.urls import path, include  # URL 패턴 함수 및 include 임포트
from myapp import views  # myapp의 views 임포트

urlpatterns = [
    path("admin/", admin.site.urls),  # /admin/ 요청 시 관리자 사이트로 연결

    path("", views.index, name="index"),  # 루트 URL 요청 시 index 뷰 호출
    path("show/", include('myapp.urls')),  # /show/로 시작하는 URL은 myapp의 urls.py에 위임
]
