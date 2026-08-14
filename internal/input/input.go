package input

import (
	"bufio"
	"context"
	"os"
	"strings"
)

// ReadStdinInputs adapts stdin lines into agent inputs, one per line. It
// returns when stdin closes (e.g. Ctrl-D) or ctx is cancelled. A blocking
// stdin read can't be interrupted, so on cancellation the reader goroutine
// may leak until the process exits; that's acceptable since the CLI is
// shutting down anyway.
func ReadStdinInputs(ctx context.Context, out chan<- string) {
	lines := make(chan string)
	go func() {
		defer close(lines)
		reader := bufio.NewReader(os.Stdin)
		for {
			line, err := reader.ReadString('\n')
			line = strings.TrimSpace(line)
			if line != "" {
				select {
				case lines <- line:
				case <-ctx.Done():
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()
	for {
		select {
		case <-ctx.Done():
			return
		case line, ok := <-lines:
			if !ok {
				return
			}
			select {
			case <-ctx.Done():
				return
			case out <- line:
			}
		}
	}
}
