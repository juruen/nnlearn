.PHONY: audit

audit:
	golangci-lint run ./...
