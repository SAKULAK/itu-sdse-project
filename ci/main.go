package pipeline

import (
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

// Training step function. Load data and preprocess data should be implemented in the future, so that they return a dagger.Directory that can be mounted into the container.
func (m *Pipeline) Train(ctx context.Context) error {
	// rawData := m.LoadData(key and shit maybe)
	// processedData := m.Preprocess(rawData)

	_, err := m.BaseContainer().
		WithDirectory("/src/bsops_the_final", m.Source.Directory("bsops_the_final")).
		WithDirectory("/src/data/processed", processedData).
		WithExec([]string{"python", "bsops_the_final/train.py"}).
		Sync(ctx)

	return err
}