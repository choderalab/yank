Developer Notes / Tools
=======================

Assorted notes for developers.

How to do a release
-------------------
- Update the whatsnew.rst document. Use the github view that shows all the commits to master since the last release to write it.
- Update the version number in `setup.py`, change `ISRELEASED` to `True`
- Commit to master, and [tag](https://github.com/rmcgibbo/mdtraj/releases) the release on github
- To push the source to PyPI, use `python setup.py sdist --formats=gztar,zip upload`
- Conda binaries need to built separately on each platform (`conda build mdtraj; binstar upload <path to .tar.bz2>`)
- Make an annoucement on github / email
- After tagging the release, make a NEW commit that changes `ISRELEASED` back to `False` in `setup.py`


It's important that the version which is tagged on github for the release be
the one with the ISRELEASED flag in setup.py set to true.


How to contribute changes
-------------------------
- Clone the repository
  * We prefer feature branches for PRs instead of forks
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Push the branch to the main repo with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!


Checklist for all Updates
-------------------------
- [ ] Update `setup.py` version number (see specific update type for details)
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version as the one in `setup.py`
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in


Checklist for Major Revisions (YANK X.Y+1.0)
------------------------------------------
- [ ] Make sure all issues related to the milestone will be closed by this commit or moved to future releases
- [ ] Update `docs/changelog.rst`
- [ ] Update `setup.py` with version number and `ISRELEASED` to `True`
- [ ] Do the steps for All Upates
- [ ] Create a new release on GitHub, reference the tag and copy the changes in `docs/changelog.rst`
- [ ] Update the `omnia-md/conda-recipies` repo by creating a new PR with updated versions
- [ ] Create a new milestone for YANK X.(Y+1).0
- [ ] Close current milestone

Checklist for Minor Revisions (YANK X.Y.Z+1)
--------------------------------------------
- [ ] Update `setup.py` with the correct Z version number in X.Y.Z
- [ ] In `setup.py`, set `ISRELEASED` to `False`
- [ ] Do all the steps for All Updates

- If this is a critical bugfix (i.e. YANK X.Y.0 is broken and/or flat out wrong without the fix):
- [ ] Update `docs/changelog.rst`
- [ ] Update the released version on the site to this version, adjusting the tag and note that this is a critical bugfix which corrects the X.Y release
- [ ] Update the `omnia-md/conda-recipies` repo to point at the corrected version
