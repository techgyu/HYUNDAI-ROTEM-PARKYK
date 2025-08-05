import random
import time

# 정렬 알고리즘의 시간을 측정하는 함수
def measure_sort_time(sort_function, data, label=None, show_result=True):
    start_time = time.time()  # 시작 시간 기록
    sorted_data = sort_function(data)  # 정렬 함수 실행
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    if label is not None:
        print(f"\n{label} 실행 시간: {elapsed_time:.6f}초")
        if show_result:
            print(f"정렬된 리스트: {sorted_data[:100]}...")  # 정렬된 리스트의 처음 100개 요소만 출력
    return sorted_data, elapsed_time
# ---------- 정렬 알고리즘 구현부 ----------

# 선택 정렬(Selection Sort)은 주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 최소값 찾기: 정렬되지 않은 부분에서 가장 작은 값을 찾는다
# 2. 교환: 찾은 최소값을 정렬되지 않은 부분의 첫 번째 위치와 교환한다
# 3. 반복: 정렬된 부분을 확장하며 위 과정을 반복한다

# 선택 정렬 방법 1: 새로운 리스트(result)를 만들어 정렬 (메모리 사용 많음)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(n) (result 리스트 사용)
def find_min_value(list_arguement): # 최소값 찾기 함수
    length = len(list_arguement)
    index = 0
    for i in range(1, length):
        if list_arguement[i] < list_arguement[index]:
            index = i
    return list_arguement[index]

def selection_sort1(list_arguement, verbose=False): # 선택 정렬 함수
    result = []
    while list_arguement:  # 리스트가 비어있지 않을 때까지 반복
        if verbose:
            print("\n현재 리스트 상태:", list_arguement)
        min_value = find_min_value(list_arguement)  # 최소값 찾기
        if verbose:
            print(f"찾은 최소값: {min_value}")
        result.append(min_value)  # 최소값을 결과 리스트에 추가
        if verbose:
            print(f"탐색한 최소 값을 결과 리스트에 추가: {result}")
        list_arguement.remove(min_value)  # 원래 리스트에서 최소값 제거
        if verbose:
            print(f"원래 리스트에서 최소값 제거 후: {list_arguement}")

    if verbose:
        print(f"\n최종 정렬 결과: {result}")
    return result

# 선택 정렬 방법 2: 제자리 정렬 (메모리 절약, 실제 문제 풀이에 적합)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(1) (제자리 정렬, 추가 메모리 사용 없음)
def selection_sort2(list_arguement, verbose=False): # 제자리 정렬 함수
    if verbose:
        print("선택 정렬(메모리 공간 절약):")
    length = len(list_arguement)
    if verbose:
        print(f"확인한 배열 길이: {length}")

    for i in range(length - 1):
        min_index = i
        for j in range(i + 1, length):
            if verbose:
                print(f"{list_arguement[j]}와 {list_arguement[min_index]}를 비교")
            if list_arguement[j] < list_arguement[min_index]:
                if verbose:
                    print(f"{list_arguement[j]}가 {list_arguement[min_index]}보다 작다")
                min_index = j
                if verbose:
                    print(f"\n갱신된 최소 값[{min_index}]: {list_arguement[min_index]}")
            else:
                if verbose:
                    print(f"{list_arguement[j]}는 {list_arguement[min_index]}보다 크거나 같다")
        if min_index != i:
            if verbose:
                print(f"  교환: list_arguement[{i}]({list_arguement[i]}) <-> list_arguement[{min_index}]({list_arguement[min_index]})")
            list_arguement[i], list_arguement[min_index] = list_arguement[min_index], list_arguement[i]
        else:
            if verbose:
                print("  교환 없음 (이미 최소값)")
        if verbose:
            print(f"  결과: {list_arguement}")
    if verbose:
        print(f"\n최종 정렬 결과: {list_arguement}")
    return list_arguement

# ---------- 선택 정렬 함수 끝 ----------

# -------------------------------------------------------------- 알고리즘 실행 부분 ---------------------------------------------------------------
# 테스트

test_list = random.sample(range(1, 1000), 500)  # 1부터 1000 사이의 숫자 중에서 500 개를 랜덤하게 선택하여 리스트 생성

# print("생성한 테스트 리스트:", test_list)

# 공간 복잡도가 낮을 수록 대규모 데이터 처리에 유리
# 시간 복잡도가 낮을 수록 빠른 정렬이 가능

# 선택 정렬 1번 실행 (O(n^2) 시간, O(n) 공간)
# - 새로운 리스트(result)에 최소값을 하나씩 추가하고, 원본 리스트에서 remove로 값을 제거함
# - remove 연산이 O(n)이라 실제로는 오버헤드가 큼, 원본 리스트는 소모됨(파괴적)
sorted_list, exec_time = measure_sort_time(
    lambda x: selection_sort1(x, verbose=True),
    test_list.copy(),
    label="선택 정렬 1번",
    show_result=True  # 결과 출력 여부 옵션
)

# 선택 정렬 2번 실행 (O(n^2) 시간, O(1) 공간)
# - 리스트 내부에서 인덱스끼리 값을 교환(제자리 정렬, in-place)
# - 추가 메모리 사용 없음, 원본 리스트가 직접 변경됨
sorted_list, exec_time = measure_sort_time(
    lambda x: selection_sort2(x, verbose=True),
    test_list.copy(),
    label="선택 정렬 2번",
    show_result=True
)