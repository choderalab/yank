echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" == "true" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


if [[ "2.7 3.3" =~ "$python" ]]; then
    conda install --yes binstar
    binstar -t $BINSTAR_TOKEN  upload --force -u omnia -p yank-dev $HOME/miniconda/conda-bld/linux-64/yank-dev-*
fi

if [[ "$python" != "2.7" ]]; then
    echo "No deploy on PYTHON_VERSION=${python}"; exit 0
fi

#make docs and push to s3
sudo apt-get install -qq pandoc         # notebook -> rst
conda install --yes matplotlib scikit-learn sphinx==1.2.3 boto ipython-notebook jinja2

cd docs && make html && cd -
python tools/ci/push-docs-to-s3.py
