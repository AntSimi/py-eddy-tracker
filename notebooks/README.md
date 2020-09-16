python setup.py install
python setup.py build_sphinx
rsync -vrltp doc/python_module notebooks/.   --include '*/' --include '*.ipynb' --exclude '*' --prune-empty-dirs
