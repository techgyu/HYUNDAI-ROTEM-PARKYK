from django.urls import path  # path 함수 임포트
from myapp import views  # myapp의 views 임포트

urlpatterns = [
    path("", views.show, name="show"),  # URL이 비어있을 때 show 뷰 호출
]