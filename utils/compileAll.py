import os

directories = [
    'common',
    'FlaskStream',
    'goal-depth-intake-detection-host',
    'pipelines'
]

ignore = [
    '__init__.py',
    'object_counter_script.py'
]

python_files = []
for d in directories:
    for root, dirs, files in os.walk("..\\{}".format(d)):
        for f in files:
            if f in ignore:
                continue

            if f.endswith(".py"):
                python_files.append(os.path.join(root, f))

print(python_files)
