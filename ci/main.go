package pipeline

import (
	"context"

	"dagger.io/dagger/core"
	"dagger/pipeline/internal/dagger"
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
			Include: []string{"bsops_the_final/", "tests/", "requirements*.txt", "data/", ".dvc/"},
			Exclude: []string{"**/__pycache__", "**/.pytest_cache", ".dvc/cache", ".dvc/tmp", ".dvc/config.local"},
		}),
	}
}

// Base container for the pipeline. Installs dependencies from requirements.txt and sets the working directory to /src.
func (m *Pipeline) BaseContainer() *dagger.Container {
	return core.NewContainer().
		From(m.BaseImageAddress).
		WithDirectory("/src", m.Source, dagger.ContainerDirectoryOpts{
			Include: []string{"requirements*.txt"},
		}).
		WithWorkdir("/src").
		WithExec([]string{"uv", "pip", "install", "--system", "--no-cache", "-r", "requirements.txt"})
}
