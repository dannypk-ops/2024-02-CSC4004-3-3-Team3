import argparse
from ultralytics import YOLO
import numpy as np
import cv2
import json
from collections import defaultdict
from PIL import Image
import os

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description="YOLOv8 Crack Detection")
parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights file")
parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing images")
parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder for results")
parser.add_argument("--frame_count", type=int, required=True, help="Number of frames to process")
args = parser.parse_args()

# YOLOv8 모델 로드
model = YOLO(args.weights)

# 입력 및 출력 디렉토리 설정
input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# 프레임 처리
for frame_num in range(0, args.frame_count):  
    file_name = f'frame_{frame_num:04d}.jpg'
    image_path = os.path.join(input_folder, file_name)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # YOLO 탐지 수행
    results = model(image_path)
    result = results[0]

    # 원래 이미지 이름을 기반으로 저장할 파일 이름 설정
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 바운딩 박스를 그리지 않고 균열 탐지 영역만 표시된 이미지 생성
    output_image_path = os.path.join(output_folder, f'{base_name}_detected.jpg')

    # boxes=False를 사용하여 바운딩 박스 없이 시각화
    annotated_image = result.plot(boxes=False)

    # PIL을 사용해 BGR을 RGB로 변환 후 저장
    annotated_image = Image.fromarray(annotated_image[..., ::-1])  # BGR -> RGB 변환
    annotated_image.save(output_image_path)
    print(f"Detection result image without bounding boxes saved as {output_image_path}")

    # 마스크 내부의 데이터를 JSON 파일로 저장
    if result.masks is not None:
        mask_data = defaultdict(list)

        for mask, score in zip(result.masks.data, result.boxes.conf):
            # 마스크를 numpy 배열로 변환
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # 마스크 내부의 채워진 픽셀 좌표를 가져옴
            filled_coordinates = np.argwhere(mask_np > 0)  # 값이 0보다 큰 좌표만 추출 (y, x 순서)

            # 좌표를 x, y 순서로 변환하고 Python 기본 데이터 타입으로 변환
            filled_coordinates_list = [list(map(int, coord[::-1])) for coord in filled_coordinates]

            # detection_score에 대한 데이터 추가
            mask_data[round(float(score.item()), 2)].append({
                "filled_coordinates": filled_coordinates_list
            })

        # JSON 파일로 저장 (원래 이름 기반으로)
        json_path = os.path.join(output_folder, f'{base_name}_detected.json')
        with open(json_path, 'w') as json_file:
            json.dump(mask_data, json_file, indent=4)
        print(f"Filled mask pixel data saved as {json_path}")
    else:
        print(f"No masks were detected in {image_path}.")
