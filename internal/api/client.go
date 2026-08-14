package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
)

// Client is the main API client for interacting with the chat completion API.
type Client struct {
	baseURL    *url.URL
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a new API client with the given base URL and API key.
func NewClient(baseURL *url.URL, apiKey string) Client {
	return Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: &http.Client{
			// Timeout: 5 * time.Minute, // We can't be sure how long the response is going to take, very thoughtful responses and thinking can take a very long time
		},
	}
}

// ChatCompletion issues a chat completion request to an openAI compatible endpoint
func (c Client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (_ ChatCompletionResponse, retErr error) {
	verbose := false
	if os.Getenv("VERBOSE") != "" {
		verbose = true
	}
	reqURL := c.baseURL.JoinPath("/chat/completions")
	body, err := json.Marshal(req)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("marshalling request: %w", err)
	}
	var reqBytes bytes.Buffer
	tee := io.TeeReader(bytes.NewReader(body), &reqBytes)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL.String(), tee)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("creating request with context: %w", err)
	}
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	httpReq.Header.Set("User-Agent", "mu2/0.1.0")
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("executing http request: %w", err)
	}
	if verbose {
		var prettyJSON bytes.Buffer
		err := json.Indent(&prettyJSON, reqBytes.Bytes(), "", "  ")
		// intentionally ignore error, just don't print verbose logs if there was an error marshalling the output
		if err == nil {
			fmt.Printf("\033[93mrequest:\n---\n%s\n---\033[0m\n\n", prettyJSON.String())
		}
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode < 200 || httpResp.StatusCode > 299 {
		errorMsg, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return ChatCompletionResponse{}, fmt.Errorf("reading error body: %w", err)
		}
		var prettyJSON bytes.Buffer
		err = json.Indent(&prettyJSON, reqBytes.Bytes(), "", "  ")
		switch err {
		case nil:
			fmt.Printf("failed request body:\n%s\n", prettyJSON.String())
		default:
			fmt.Printf("failed request body: %s\n", reqBytes.String())
		}
		return ChatCompletionResponse{}, fmt.Errorf("request failed [%s; code: %d]: %s response: %s", reqURL.String(), httpResp.StatusCode, errorMsg, reqBytes.String())
	}
	resp := ChatCompletionResponse{}
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("decoding response: %w", err)
	}
	if verbose {
		b, err := json.MarshalIndent(resp, "", "  ")
		// intentionally ignore error, just don't print verbose logs if there was an error marshalling the output
		if err == nil {
			fmt.Printf("\033[91mresponse:\n---\n%s\n---\033[0m\n\n", string(b))
		}
	}
	return resp, nil
}
