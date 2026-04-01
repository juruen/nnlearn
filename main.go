package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	addr := ":8080"
	if envAddr := os.Getenv("NNLEARN_ADDR"); envAddr != "" {
		addr = envAddr
	}

	fileServer := http.FileServer(http.Dir("web"))
	mux := http.NewServeMux()
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			r.URL.Path = "/index.html"
		}
		fileServer.ServeHTTP(w, r)
	}))

	log.Printf("nnlearn static web app available at http://localhost%s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
