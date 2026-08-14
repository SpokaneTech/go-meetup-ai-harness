package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
	httpResp, err := c.httpClient.Do(httpReq) //nolint:bodyclose // intentionally leaving body open so it can be used in the ChatStream
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("executing http request: %w", err)
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
	var buf bytes.Buffer
	teeBody := io.TeeReader(httpResp.Body, &buf)
	if err := json.NewDecoder(teeBody).Decode(&resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("decoding response: %w", err)
	}
	fmt.Printf("response body: %s\n", buf.String())
	return resp, nil
}
