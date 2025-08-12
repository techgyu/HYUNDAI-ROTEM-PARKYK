from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import seaborn as sns
import matplotlib

# GUI 환경 없이, 이미지 파일로 저장하기 위해 backend를 설정합니다.
matplotlib.use('Agg') # matplotlib이 그래프를 그릴 때 backend로 지정하는 코드
# 기본적으로 그래프 창이 열리는데, Agg를 사용하면 GUI 창 없이 이미지 파일로 저장할 수 있습니다.
# Agg는 pdf, svg, png 등의 포맷으로 이미지를 저장할 수 있습니다.

import matplotlib.pyplot as plt

# Create your views here.
def main(request):
    return render(request, "main.html")

def showdata(request):
    # data 로딩
    df = sns.load_dataset('iris')

    # image 저장 경로 설정 <BASE_DIR?/static/images/iris.png>
    static_app_dir = Path(settings.BASE_DIR) / "static" / "images"
    static_app_dir.mkdir(parents=True, exist_ok=True)
    img_path = static_app_dir / "iris.png"

    # 산점도(차트) 저장
    counts = df['species'].value_counts().sort_index()
    print("counts: ", counts)

    plt.figure()
    counts.plot.pie(autopct='%1.1f%%', startangle=90)  # type: ignore
    plt.ylabel('')  # 필요하다면 빈 y축 라벨 추가
    plt.title("Iris Species Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(img_path, dpi=130)
    plt.close()

    # df를 table tag로 만들어서 show.html에 전달
    table_html = df.to_html(classes='table table-striped table-sm', index=False)
    
    return render(request, "show.html", {
        'table': table_html,
        "img_relpath": 'images/iris.png'
        })