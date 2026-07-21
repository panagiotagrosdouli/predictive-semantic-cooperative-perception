# Dataset preparation

## OPV2V

OPV2V is the primary dataset for the first experiments. It is an open, simulated vehicle-to-vehicle cooperative-perception benchmark distributed through the official OpenCOOD project.

Official resources:

- OpenCOOD repository: `https://github.com/DerrickXuNu/OpenCOOD`
- Data preparation guide: `https://opencood.readthedocs.io/en/latest/md_files/data_intro.html`
- OPV2V project page: `https://mobility-lab.seas.ucla.edu/opv2v/`

## Expected layout

After downloading and extracting the official archives, place the data under:

```text
datasets/opv2v/
├── train/
├── validate/
├── test/
└── test_culvercity/
```

The raw dataset must not be committed to GitHub. The repository `.gitignore` excludes it.

## Download

Run:

```bash
bash scripts/download_opv2v.sh
```

The script opens the official instructions because the maintainers distribute the dataset through external cloud-hosted archives whose direct file identifiers may change. Download either the complete archives or the official chunked archives.

For chunked downloads, reconstruct and extract each split with:

```bash
cat train.zip.part* > train.zip
unzip train.zip -d datasets/opv2v
```

Repeat for validation and test splits as needed.

## First development stage

The repository demo does not require OPV2V. It uses deterministic synthetic tracks to test prediction, risk estimation and semantic scheduling before the dataset adapter is implemented.

## Licence and citation

Use OPV2V/OpenCOOD only under their published terms and cite the original OPV2V paper in research outputs.