@echo off
cd "C:\Users\kissa zahra\Desktop\semesters\Semester 7\MLOps\MLOPs Final Project\course-project-kissasium"
call "C:\Users\kissa zahra\Desktop\semesters\Semester 7\MLOps\MLOPs Final Project\course-project-kissasium\venv\Scripts\activate.bat"
py "C:\Users\kissa zahra\Desktop\semesters\Semester 7\MLOps\MLOPs Final Project\course-project-kissasium\code\data_collections.py"
dvc add data\weather_data.csv data\air_quality_data.csv
git add .
git commit -m "Update data files"
dvc push
git push
