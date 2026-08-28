# Releasing rlaopt

This document is the authoritative release checklist for maintainers. rlaopt uses a static version in `pyproject.toml`, uv for building and validation, GitHub Releases for release notes, and PyPI Trusted Publishing for deployment. Do not publish from a maintainer workstation.

## One-time repository setup

Complete these steps before the first release:

1. In the GitHub repository, create an environment named `pypi` under **Settings → Environments**.
   - Add at least one required reviewer so publishing requires explicit approval.
   - Restrict deployments to protected tags matching `v*`.
   - Do not add a PyPI token or other publishing secret.
2. Add a GitHub tag ruleset for `v*` that prevents unauthorized tag creation, updates, and deletion. Allow only the maintainers who release the package to bypass it.
3. In the PyPI account that will own the project, open **Publishing** and add a pending GitHub Trusted Publisher with these exact values:
   - PyPI project name: `rlaopt`
   - GitHub owner: `udellgroup`
   - GitHub repository: `rlaopt`
   - Workflow name: `publish-release.yml`
   - Environment name: `pypi`

A pending publisher creates the PyPI project on its first successful use; it does not reserve the name beforehand. After the first release, add at least one other maintainer as a PyPI owner.

For Read the Docs, activate the first released tag after it appears and set the stable release as the default version. Keep `latest` available for documentation built from `main`.

## Versioning

Use [PEP 440](https://peps.python.org/pep-0440/) versions and choose increments according to [Semantic Versioning](https://semver.org/). While rlaopt is in the `0.x` series, a minor release may contain breaking API changes, but those changes must be called out prominently in the release notes.

`[project].version` in `pyproject.toml` is the only version maintainers edit. The installed `rlaopt.__version__` and the Sphinx documentation version are derived from package metadata.

Use uv to update the version and lockfile together:

```bash
uv version X.Y.Z --no-sync
```

Examples of valid release versions and tags include `0.2.0` / `v0.2.0` and `0.2.0rc1` / `v0.2.0rc1`. Never reuse a version or move a published release tag.

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

Retain GitHub's generated pull-request list, contributor list, and full comparison link below those sections. For the first release, write a concise manual overview because there is no previous release from which GitHub can generate a comparison.

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

1. Check the project page and provenance on `https://pypi.org/project/rlaopt/`.
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
