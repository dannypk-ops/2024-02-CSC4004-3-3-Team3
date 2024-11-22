#!/bin/bash

# COLMAP 프로젝트 폴더 설정
# 실제로는 Network로부터 받은 경로로 할당을 해야 한다.
PROJECT_DIR="/home/dannypk99/Desktop/Colmap/Testing"
IMAGE_PATH="/home/dannypk99/Desktop/dataset/datasets/Crack/building"
DB_PATH="$PROJECT_DIR/database.db"
SPARSE_PATH="$PROJECT_DIR/sparse"
JSON_PATH="$PROJECT_DIR/detected_results"
RESULT_PATH = "$PROJECT_DIR/ply_output"

# 2D Image에 대한 crack detection 수행후, JSON 파일로 저장.


# 프로젝트 디렉터리 및 관련 폴더가 없을 경우 생성
mkdir -p "$PROJECT_DIR"
mkdir -p "$IMAGE_PATH"
mkdir -p "$SPARSE_PATH"
mkdir -p "$JSON_PATH"


#!/bin/bash

# YOLO 가중치 파일 경로
WEIGHTS="weights/best.pt"
OUTPUT_FOLDER= "$JSON_PATH"
FRAME_COUNT=223

# Python 스크립트 실행
python detect_cracks.py --weights "$WEIGHTS" --input_folder "$IMAGE_PATH" --output_folder "$OUTPUT_FOLDER" --frame_count "$FRAME_COUNT"


# 데이터베이스 파일이 없으면 새로 생성
if [ ! -f "$DB_PATH" ]; then
    echo "Creating new database at $DB_PATH"
    sqlite3 "$DB_PATH" "VACUUM;"
fi

# 데이터베이스 생성 및 기능 추출
colmap feature_extractor \
    --database_path $DB_PATH \
    --image_path $IMAGE_PATH \
    --ImageReader.camera_model "SIMPLE_PINHOLE" \
    --ImageReader.single_camera 1

# 특징 매칭
colmap exhaustive_matcher \
    --database_path $DB_PATH

# SfM 재구성
colmap mapper \
    --database_path $DB_PATH \
    --image_path $IMAGE_PATH \
    --output_path $SPARSE_PATH

python train.py \
    --weights "$WEIGHTS" \
    --detected_results "$JSON_PATH" \
    -s $SPARSE_PATH\
    --save_path "$RESULT_PATH"