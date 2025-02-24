#!/bin/bash
TARGET_DIR="../videos/"
find "$TARGET_DIR" -type d -empty -delete
echo "所有视频空目录已删除。"
