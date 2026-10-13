@echo off
title Running Spina Bifida System Unit Tests
echo ========================================================
echo Running Pytest Suite...
echo ========================================================
python -m pytest tests/test_pipeline.py -v
pause
