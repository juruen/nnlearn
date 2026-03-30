.PHONY: audit test

audit:
	golangci-lint run ./...

test:
	go test ./... -v
