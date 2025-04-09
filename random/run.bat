@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Running Adversarial Knowledge Distillation implementation...
python main.py

echo Done! 