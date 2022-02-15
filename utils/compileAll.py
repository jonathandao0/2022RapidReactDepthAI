import os

directories = [
    'common',
    'FlaskStream',
    'goal-depth-intake-detection-host',
    'pipelines'
]

python_files = []
for d in directories:
    for root, d, files in os.walk("..\\{}".format(d)):
        for f in files:
            if f == '__init__.py':
                continue

            if f.endswith(".py"):
                python_files.append(os.path.join(root, f))

print(python_files)
