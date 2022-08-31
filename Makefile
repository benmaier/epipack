PKG=epipack

default: 
	make python

clean:
	-rm -f *.o
	make pyclean

clean_all:
	make clean
	make pyclean

pyclean:
	-rm -f *.so
	-rm -rf *.egg-info*
	-rm -rf ./tmp/
	-rm -rf ./build/

python:
	pip install -e ../${PKG}

checkdocs:
	python setup.py checkdocs

pypi:
	mkdir -p dist
	touch dist/foobar
	rm dist/*
	python setup.py sdist
	twine check dist/*

upload:
	twine upload dist/*

readme:
	pandoc --from gfm --to rst README.md > _README.rst
	sed -e "s/^\:\:/\.\. code\:\: bash/g" _README.rst > README.rst
	rm _README.rst
	rstcheck README.rst
	cd docs; python make_about.py

test:
	pytest --cov=${PKG} ${PKG}/tests/

authors:
	python authorlist.py

grootinstall:
	/opt/python36/bin/pip3.6 install --user ../${PKG}

groot:
	git fetch
	git pull
	make grootinstall
