This guide outlines a step by step process for hosting the project on azure once you have a working docker image on ghcr.io
ANYTHING MARKED WITH <text> needs to be replaced by your own text. 

Install az cli tool
MacOs:
```console
brew install azure-cli
```

Linux/WSL:
```console
curl -fsSL 'https://azurecliprod.blob.core.windows.net/$root/deb_install.sh' | sudo bash
```


Login (one-time setup)

Creates a service principal scoped to the resource group, registers the providers, and writes
its credentials to `.env.azure` (gitignored). Opens a browser login if you aren't logged in yet.
Run from Git Bash on Windows.
```console
./scripts/azure/bootstrap-sp.sh
```

Login (every time after that, no browser)

Uses `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` from the
environment (e.g. CI secrets), falling back to `.env.azure`.
```console
./scripts/azure/login.sh
```

Deploy (every release, after the image is pushed to ghcr.io)

Logs in via `login.sh`, creates the Container Apps environment if it doesn't exist yet, fills the
placeholders in `infra/azure/app.yaml`, and creates or updates the app. Safe to re-run.
The GHCR package must be public so Azure can pull it.
```console
./scripts/azure/deploy.sh <your-github-user> <your-image-tag>
./scripts/azure/deploy.sh <your-github-user> <your-image-tag> --dry-run   # just print the rendered spec
```

Defaults: resource group `mlops-itu-rg`, region `norwayeast`, environment `mlops-itu-env`,
app `mlops-itu-jtk`. Override with `AZ_RG`, `AZ_LOC`, `AZ_ENV`, `AZ_APP`.

In CI (GitHub Actions), store `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID` as repository secrets, expose them as env vars, and run the same command.
