import os
import shutil
from yank import version

if version.release:
    docversion = version.version
else:
    docversion = 'development'

try:
    os.mkdir("docs/_deploy")
except:
    pass

shutil.copytree("docs/_build", "docs/_deploy/{docversion}"
                .format(docversion=docversion))

# Only update latest if we are on a release version
if version.release:
    # Copy the most recent version to the latest build
    # copytree will make the directory for us
    shutil.copytree("docs/_build", "docs/_deploy/latest"
                    .format(docversion=docversion))
