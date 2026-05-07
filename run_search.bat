@echo off
cd /d %~dp0
call .venv\Scripts\activate

python app.py --load-index --query "what is customer relationship management" --retriever dense --generate-answer --backend local
pause