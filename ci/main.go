package pipeline

import (
	"context"
	"dagger/pipeline/internal/dagger"

	"dagger.io/dagger/core"
)

type Pipeline struct {
	// Image the build runs on.
	BaseImageAddress string

	// +private
	Source *dagger.Directory
}

// Constructor for a new Pipeline. Defines the base image and source directory.
func New(
	ws *dagger.Workspace,
	// +default="ghcr.io/astral-sh/uv:python3.13-trixie-slim"
	baseImageAddress string,
) *Pipeline {
	return &Pipeline{
		BaseImageAddress: baseImageAddress,
		Source: ws.Directory("/", dagger.WorkspaceDirectoryOpts{
			Gitignore: true,
			Include:   []string{"bsops_the_final/", "tests/", "requirements*.txt", "data/", ".dvc/"},
			Exclude:   []string{"**/__pycache__", "**/.pytest_cache", ".dvc/cache", ".dvc/tmp", ".dvc/config.local"},
		}),
	}
}

// Base container for the pipeline. Installs dependencies from requirements.txt and sets the working directory to /src.
func (m *Pipeline) Container() *dagger.Container {
	return core.NewContainer().
		From(m.BaseImageAddress).
		WithDirectory("/src", m.Source, core.ContainerWithDirectoryOpts{
			Include: []string{"requirements*.txt"},
		}).
		WithWorkdir("/src").
		WithExec([]string{"uv", "pip", "install", "--system", "--no-cache", "-r", "requirements.txt"})
}

// Load data from DVC. Assumes that the data is stored in a DVC repository and that the DVC remote is configured.
func (m *Pipeline) LoadData() *dagger.Directory {
	container := m.Container().
		WithDirectory("/src/bsops_the_final", m.Source.Directory("bsops_the_final")).
		WithDirectory("/src/data", m.Source.Directory("data")).
		WithDirectory("/src/.dvc", m.Source.Directory(".dvc")).
		WithExec([]string{"dvc", "config", "--local", "core.no_scm", "true"}).
		WithExec([]string{"dvc", "pull",
			"data/images/metadata.csv.dvc",
			"data/test/metadata.csv.dvc",
			"-v"}).
		WithExec([]string{"python", "-c",
			"from bsops_the_final.data.data import pull_data_from_dvc; pull_data_from_dvc()"})

	return container.Directory("/src/data")
}

// Preprocess data by performing transformations, splitting it into train, validation, and test sets, and saving the processed data to data/processed.
// comment by Zita: couldn't test it yet. The data/processed directory will be created in the container and will contain train.csv, val.csv, and test.csv files with the paths and labels of the images.
func (m *Pipeline) PreprocessData() *dagger.Directory {
	container := m.Container().
		WithDirectory("/src/bsops_the_final", m.Source.Directory("bsops_the_final")).
		WithDirectory("/src/data", m.LoadData()).
		WithWorkdir("/src").
		// dataset.py only defines functions (no __main__ entry point) and nothing
		// in the provided Python writes the splits to disk. Instead of changing the original
		// Python code, this inline script calls it and saves train/val/test
		// as CSVs so they can be exported as a Dagger directory.
		// (Zita) I'm a little unsure about this part, if you have a better idea of how to do it,
		// please let me know.
		WithExec([]string{"python", "-c", `
import csv
from pathlib import Path
from bsops_the_final.data.data import load_image_dataset
from bsops_the_final.data.dataset import split_dataset

train, val, test = split_dataset(load_image_dataset())

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
for name, rows in [("train", train), ("val", val), ("test", test)]:
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows((str(p), l) for p, l in rows)
`})

	return container.Directory("/src/data/processed")
}

// Training step function. Load data and preprocess data should be implemented in the future, so that they return a dagger.Directory that can be mounted into the container.
func (m *Pipeline) Train(ctx context.Context) error {
	// rawData := m.LoadData(key and shit maybe)
	// processedData := m.Preprocess(rawData)

	_, err := m.Container().
		WithDirectory("/src/bsops_the_final", m.Source.Directory("bsops_the_final")).
		// WithDirectory("/src/data/processed", processedData).
		WithExec([]string{"python", "bsops_the_final/train.py"}).
		Sync(ctx)

	return err
}
