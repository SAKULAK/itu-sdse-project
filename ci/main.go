package pipeline

import (
	"dagger.io/dagger/core"

	"dagger/pipeline/internal/dagger"
)

type Pipeline struct {
	// Image the build runs on.
	BaseImageAddress string

	// +private
	Source *dagger.Directory
}

func New(
	ws *dagger.Workspace,
	// +default="alpine:3.21"
	baseImageAddress string,
) *Pipeline {
	return &Pipeline{
		BaseImageAddress: baseImageAddress,
		Source: ws.Directory("/", dagger.WorkspaceDirectoryOpts{
			Exclude: []string{"**/.git", "**/.dagger", "**/vendor"},
		}),
	}
}

// A container with the source mounted, ready to build on.
func (m *Pipeline) Container() *dagger.Container {
	return core.NewContainer().
		From(m.BaseImageAddress).
		WithDirectory("/src", m.Source).
		WithWorkdir("/src")
}
