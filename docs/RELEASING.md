# Releasing AnaHD

The installer is built on GitHub's Windows runner by `.github/workflows/release.yml`
when a version tag is pushed; it ends as a **draft** release, published by hand after a
check. Nothing in this needs a local GPU.

## Once per model: the model release

The model is not in git. The installer build (and source installs without a cached model)
download it from a release, checked against `MODEL_SHA256` in `src/anahd/model/weights.py`.

1. Make the shipped file from a training checkpoint (drops the optimizer state, FP16):

   ```
   uv run python tools/export_model.py <run>/v6_final.pt anahd-v6.pt
   ```

   It prints the SHA-256; it must equal `MODEL_SHA256` (update both `MODEL_NAME` and
   `MODEL_SHA256` for a new model, and the tag in `MODEL_URL`).
2. Publish it as its own release (tag `model-v6`, matching `MODEL_URL`):

   ```
   gh release create model-v6 anahd-v6.pt --title "AnaHD model v6" \
     --notes "Weights for the Learned cleaning (MIT). Used by the installer and source installs."
   ```

## Each version

1. Set the version in `pyproject.toml` and `src/anahd/__init__.py` (they must match the tag).
2. Write `docs/release-notes/v<version>.md` (the release text; the workflow adds the
   installer's SHA-256) and add the version to `CHANGELOG.md`.
3. Run the full tests on a GPU machine: `uv run pytest`.
4. Tag and push: `git tag v<version> && git push origin v<version>`.
5. The workflow (about an hour) builds and checks the installer, fails if it is 2 GiB or
   larger (GitHub's limit for a release file), and creates the draft release.
6. Download the installer from the draft, install it, clean a clip, then publish the
   release.

To build an installer without releasing, run the workflow by hand (Actions > Release >
Run workflow): the installer is then a workflow artifact for 7 days.

## Locally

```
uv sync --extra app --group build
uv run python scripts/build_windows.py --ffmpeg <ffmpeg bin folder>
iscc /DAppVersion=<version> packaging\installer.iss
```

The installer lands in `dist\`. The repository must be public for anyone to download a
release.
