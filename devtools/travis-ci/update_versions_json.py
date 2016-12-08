import json

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from yank import version

if not version.release:
    print("This is not a release.")
    exit(0)

URL = 'http://www.getyank.org'
try:
    data = urlopen(URL + '/versions.json').read().decode()
except:
    data = '[\n]\n'
versions = json.loads(data)

# Debug lines
# import pdb
# sd = urlopen('http://mdtraj.org' + '/versions.json').read().decode()
# sv = json.loads(sd)

# Sort the list so the versions are in the right order online
versions = sorted(versions, key=lambda k: k['version'])

# new release so all the others are now old
for i in range(len(versions)):
    versions[i]['latest'] = False

versions.append({
    'version': version.version,
    'display': version.short_version,
    'url': "{base}/{version}".format(base=URL, version=version.version),
    'latest': True,
})

with open("docs/_deploy/versions.json", 'w') as versionf:
    json.dump(versions, versionf, indent=2)
