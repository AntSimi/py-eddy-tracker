isort src tests examples
black src tests examples
blackdoc src tests examples
flake8 tests examples src --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 tests examples src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
pytest -vv --cov py_eddy_tracker --cov-report html
