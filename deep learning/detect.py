from ultralytics import YOLO
import cv2
import json
import os

# YOLOv8 모델 로드 (사용자 지정 가중치 경로)
model = YOLO('C:/crack/weights/best.pt')

# 이미지 파일이 있는 디렉토리 및 출력 디렉토리 설정
input_folder = 'C:/crack/building'
output_folder = 'C:/crack/detected_images'
json_folder = 'C:/crack/json_results'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

for frame_num in range(0, 223):
    file_name = f'frame_{frame_num:04d}.jpg'
    image_path = os.path.join(input_folder, file_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # YOLO 탐지 수행
    results = model(image_path)
    result = results[0]

    # 원본 이미지 이름 기반으로 저장할 JSON 데이터 구조 생성
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 탐지 데이터 저장 딕셔너리 초기화
    detected_data = {}

    if result.boxes is not None:
        for box, score, label_idx in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            # 바운딩 박스 좌표 및 기타 데이터 가져오기
            x_min, y_min, x_max, y_max = map(float, box.tolist())
            confidence = round(float(score), 4)  # 탐지 확률 (소수점 4자리까지)
            confidence_key = str(confidence)  # 확률을 키로 사용
            label = model.names[int(label_idx)]  # 클래스 레이블 이름 가져오기

            filled_coordinates = [x_min, y_min, x_max, y_max]

            # 확률 키가 없으면 초기화
            if confidence_key not in detected_data:
                detected_data[confidence_key] = []

            # 변환된 데이터 추가
            detected_data[confidence_key].append({
                "filled_coordinates": filled_coordinates,
                "label": label
            })

    # JSON 파일로 저장
    json_path = os.path.join(json_folder, f'{base_name}.json')
    with open(json_path, 'w') as json_file:
        json.dump(detected_data, json_file, indent=4)
        print(f"Detection results saved to {json_path}")

    # 결과 이미지 생성 및 저장
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    overlay = image.copy()

    for confidence_key, detections in detected_data.items():
        for detection in detections:
            x_min, y_min, x_max, y_max = map(int, detection["filled_coordinates"])

            # 바운딩 박스 영역을 반투명 파란색으로 채우기
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1) 

    # 원본 이미지와 오버레이를 합성하여 투명도 적용
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 결과 이미지 저장
    output_image_path = os.path.join(output_folder, f'{base_name}.jpg')
    cv2.imwrite(output_image_path, image)
    print(f"이미지 파일 저장 경로 {output_image_path}")
