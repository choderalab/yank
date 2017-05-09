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

# Only update latest if we are on a release version
if version.release:
    # Update the "latest" index file
    base_index_string = """<html><head><meta http-equiv="refresh" content="0;URL='/{WHEREISLATEST}'"/></head></html>"""
    index_html = base_index_string.format(WHEREISLATEST=docversion)
    try:
        os.mkdir("docs/_deploy/latest")
    except:
        pass
    with open("docs/_deploy/latest/index.html", 'w') as index_file:
        index_file.write(index_html)
