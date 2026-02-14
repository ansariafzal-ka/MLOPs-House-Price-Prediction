import os
from pathlib import Path

project_name = 'src'

list_of_files = [
    f'{project_name}/__init__.py',
    f'{project_name}/components/__init__.py',
    f'{project_name}/components/data_ingestion.py',
    f'{project_name}/components/data_preprocessing.py',
    f'{project_name}/components/model_trainer.py',
    f'{project_name}/components/model_evaluation.py',
    f'{project_name}/configurations/__init__.py',
    f'{project_name}/configurations/aws_connection.py',
    f'{project_name}/notebooks/eda.ipynb',
    f'{project_name}/notebooks/preprocessing.ipynb',
    f'{project_name}/notebooks/model_training.ipynb',
    f'{project_name}/notebooks/model_evaluation.ipynb',
    f'{project_name}/utils/__init__.py',
    f'{project_name}/logger.py',
    f'{project_name}/exception.py',
    f'{project_name}/pipeline.py',
    'app.py',
    'requirements.txt',
    'Dockerfile',
    '.dockerignore',
    '.gitignore',
    'setup.py',
    '.env'
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != '':
        os.makedirs(file_dir, exist_ok=True)
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:
            pass
    else:
        print(f'file already exists at: {file_path}')