import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU 이름: {details.get('device_name', '알 수 없음')}")
else:
    print("GPU를 찾을 수 없습니다.")