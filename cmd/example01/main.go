package main

import (
	"context"
	"fmt"
	"net/url"
	"os"

	"github.com/SpokaneText/go-meetup-ai-harness/internal/api"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENCODE_API_KEY")
	if apiKey == "" {
		panic("OPENAI_API_KEY not set")
	}
	baseURLEnv := os.Getenv("OPENCODE_BASE_URL")
	if baseURLEnv == "" {
		baseURLEnv = "https://opencode.ai/zen/go/v1"
	}
	if err := run(ctx, apiKey, baseURLEnv); err != nil {
		panic(fmt.Sprintf("run error: %v", err))
	}
}

func run(ctx context.Context, apiKey, baseURLEnv string) error {
	baseURL, err := url.Parse(baseURLEnv)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}
	client := api.NewClient(baseURL, apiKey)
	prompt := "Count to 10"
	resp, err := client.ChatCompletion(ctx, api.ChatCompletionRequest{
		Model: "deepseek-v4-flash",
		Messages: []api.Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("chat completion error: %w", err)
	}
	fmt.Println("\033[2m------------------------------------------------------------")
	fmt.Printf("ID: %s\n", resp.ID)
	fmt.Printf("Object: %s\n", resp.Object)
	fmt.Printf("Created: %d\n", resp.Created)
	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Usage - prompt tokens: %d; completion tokens: %d; total tokens: %d\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	fmt.Printf("------------------------------------------------------------\033[0m\n\n")
	fmt.Printf("Prompt:\t\t%s\n", prompt)
	if len(resp.Choices) > 0 {
		fmt.Printf("Reasoning:\t%s\n", resp.Choices[0].Message.ReasoningContent)
		fmt.Printf("Response:\t%s\n", resp.Choices[0].Message.Content)
		fmt.Printf("FinishReason:\t%s\n", resp.Choices[0].FinishReason)
	}
	return nil
}
