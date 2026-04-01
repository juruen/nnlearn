.PHONY: audit test

audit:
	golangci-lint run ./...
	go test ./... -v

test:
	go test ./... -v
