import os
import shutil
from yank import version

if version.release:
    docversion = version.version
else:
    docversion = 'development'

os.mkdir("docs/_deploy")
shutil.copytree("docs/_build", "docs/_deploy/{docversion}"
                .format(docversion=docversion))