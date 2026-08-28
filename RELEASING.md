# Releasing rlaopt

This document is the release checklist for maintainers. rlaopt uses a static version in `pyproject.toml`, uv for building and validation, GitHub Releases for release notes, and PyPI Trusted Publishing for deployment.

## Versioning

Use [PEP 440](https://peps.python.org/pep-0440/) versions and choose increments according to [Semantic Versioning](https://semver.org/).

`[project].version` in `pyproject.toml` is the only version maintainers edit. The installed `rlaopt.__version__` and the Sphinx documentation version are derived from package metadata.

Use uv to update the version and lockfile together:

```bash
uv version X.Y.Z --no-sync
```

An example of a valid release version and tag is `0.2.0` / `v0.2.0`.

## Prepare a release

1. Start from an up-to-date `main` and create a short-lived release branch:

   ```bash
   git switch main
   git pull --ff-only
   git switch -c release/X.Y.Z
   ```

2. Update the version and review both files changed by uv:

   ```bash
   uv version X.Y.Z --no-sync
   git diff -- pyproject.toml uv.lock
   ```

3. Update user-facing documentation for any changed installation, compatibility, or migration instructions.

4. Run the local packaging checks:

   ```bash
   uv lock --check
   uv build --no-sources
   uv sync --locked --group release
   uv run --locked --no-sync twine check --strict dist/*
   ```

5. Commit the release preparation, open a pull request, and merge it after CI passes. The package checks in CI rebuild the project, validate its metadata, smoke-test independent installs from both the wheel and source distribution, and build the documentation.

The release branch prepares the version; it never triggers a deployment.

## Write the release notes

GitHub Release notes are the canonical changelog. On the new release form, choose **Generate release notes**, then edit the generated content so the top contains the information users need:

```markdown
## Highlights

- The most important user-visible changes.

## Breaking changes and migrations

- Required code or configuration changes, or "None."

## Compatibility and installation

- Supported Python changes, dependency constraints, and installation notes.
```

Retain GitHub's generated pull-request list, contributor list, and full comparison link below those sections.

## Publish a release

1. Confirm the version commit is on `main` and all required checks have passed.
2. In GitHub, draft a new release targeting that exact commit on `main`.
3. Create the tag `vX.Y.Z` and use the same value as the release title.
4. Generate and edit the release notes as described above.
5. Mark the GitHub release as a prerelease if and only if the version is a PEP 440 prerelease such as `0.2.0rc1`.
6. Publish the GitHub Release. This is the irreversible deployment trigger.
7. Approve the `pypi` environment deployment after checking the tag, commit, and version shown in the workflow run.

The release workflow verifies that the tag matches `pyproject.toml`, the prerelease flag matches the version, and the tagged commit belongs to `main`. It then builds and smoke-tests the distributions in an unprivileged job. A separate job obtains a short-lived OIDC credential, generates PEP 740 attestations, and publishes the exact artifacts to PyPI. After publication, the wheel and source distribution are attached to the GitHub Release.

## Verify the release

After the workflow completes:

1. Check the project page on `https://pypi.org/project/rlaopt/`.
2. Test a clean install from PyPI:

   ```bash
   uv run --isolated --no-project --with rlaopt==X.Y.Z \
     python -c 'import rlaopt; print(rlaopt.__version__)'
   ```

3. Confirm the versioned documentation builds on Read the Docs. Activate the version if an automation rule has not done so.
4. Confirm the wheel and source distribution are attached to the GitHub Release.

## Recover from a failure

- If a transient infrastructure failure occurs before PyPI accepts any files, rerun the unchanged workflow.
- If the tagged source fails validation, do not move or reuse the tag. Fix the problem on `main`, increment the version, and publish a new release.
- If publishing is interrupted after PyPI accepts only some files, rerun the same workflow. `uv publish` safely skips identical files already present on PyPI.
- PyPI artifacts cannot be replaced. If a bad release was published, yank it when appropriate, fix the problem, increment the version, and publish a new release.
- Never delete and recreate a released tag or attempt to upload different artifacts for an existing version.
