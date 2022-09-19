isort .
black .
blackdoc .
flake8 .
python -m pytest -vv --cov py_eddy_tracker --cov-report html
