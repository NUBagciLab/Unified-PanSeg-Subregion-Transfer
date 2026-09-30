# Project website

The project page is in `docs/`. It is a self-contained static site: HTML, CSS, JavaScript, and two existing project figures. No build tools or packages are required.

## Preview

Open `docs/index.html`, or run `python -m http.server 8000 --directory docs` from the repository root and visit http://localhost:8000.

## Publish with GitHub Pages

1. Commit and push the `docs/` folder to the repository's default branch.
2. In the GitHub repository, open **Settings → Pages**.
3. Choose **Deploy from a branch**, select the default branch and **/docs**, then save.
4. After GitHub finishes publishing, the expected URL is:
   https://nubagcilab.github.io/Unified-PanSeg-Subregion-Transfer/

This file does not mean deployment has been performed.

## Complete the dataset release

Edit `docs/index.html` when these details are confirmed:

- Dataset name and released scan/patient counts; modality and center breakdown.
- Patient-level train/validation/test splits and annotation protocol.
- Final label convention, file layout, and metadata documentation.
- Download or access-request URL, dataset license, and access terms.
- Formal accepted venue and publication citation, if desired. The current citation uses the verified arXiv record.

The page deliberately marks the dataset as in preparation. The 4,604 scans refer to the paper's representation-learning cohort, not to the public subregion release. The published metrics are taken from the paper's abstract. Label values come from `Model/dataset.json`. Dataset configuration files containing case lists are not copied into the website.

To enable dataset download, replace the Dataset card's “In preparation” status with a real link and update the hero release note and Dataset section together. The Google Drive link is for model weights only.

## Edit assets

- `docs/assets/style.css`: styles and mobile layout.
- `docs/assets/site.js`: citation-copy interaction, with manual-copy fallback.
- `docs/assets/workflow.png`, `docs/assets/segmentation.png`: repository figures.

The page follows the reference site's academic layout, with an independently implemented design and a footer credit. It does not inherit or assign a dataset license from the reference website.
