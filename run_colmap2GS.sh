#!/bin/bash

# COLMAP 프로젝트 폴더 설정
# 실제로는 Network로부터 받은 경로로 할당을 해야 한다.
PROJECT_DIR="/home/dannypk99/Desktop/Colmap/Testing"
IMAGE_PATH="/home/dannypk99/Desktop/dataset/datasets/Crack/building"
DB_PATH="$PROJECT_DIR/database.db"
SPARSE_PATH="$PROJECT_DIR/sparse"
DENSE_PATH="$PROJECT_DIR/dense"

# 프로젝트 디렉터리 및 관련 폴더가 없을 경우 생성
mkdir -p "$PROJECT_DIR"
mkdir -p "$IMAGE_PATH"
mkdir -p "$SPARSE_PATH"
mkdir -p "$DENSE_PATH"

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

# Gaussian Splatting 
python train.py -s $SPARSE_PATH
