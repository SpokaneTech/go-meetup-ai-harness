package main

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"os/signal"

	"github.com/SpokaneText/go-meetup-ai-harness/internal/api"
	"github.com/SpokaneText/go-meetup-ai-harness/internal/input"
	"golang.org/x/sync/errgroup"
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
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, os.Kill)
	defer cancel()
	baseURL, err := url.Parse(baseURLEnv)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}
	client := api.NewClient(baseURL, apiKey)
	messages := []api.Message{{
		Role:    api.RoleSystem,
		Content: "You are a helpful assistant. Respond like the robot you are.",
	}}
	wg, ctx := errgroup.WithContext(ctx)
	inputCh := make(chan string)
	wg.Go(func() error {
		defer close(inputCh)
		input.ReadStdinInputs(ctx, inputCh)
		return nil
	})
	wg.Go(func() error {
		fmt.Print("> ")
		for prompt := range inputCh {
			messages = append(messages, api.Message{
				Role:    api.RoleUser,
				Content: prompt,
			})
			resp, err := client.ChatCompletion(ctx, api.ChatCompletionRequest{
				Model:    "deepseek-v4-flash",
				Messages: messages,
			})
			if err != nil {
				return fmt.Errorf("chat completion error: %w", err)
			}
			if len(resp.Choices) == 0 {
				fmt.Println("expected at least one choice, got 0")
				continue
			}
			choice := resp.Choices[0]
			printResponse(resp, choice)
			messages = append(messages, api.Message{
				Role:    api.RoleAssistant,
				Content: choice.Message.Content,
			})
			fmt.Print("> ")
		}
		return nil
	})
	return wg.Wait()
}

func printResponse(resp api.ChatCompletionResponse, choice api.Choice) {
	finishReason := choice.FinishReason
	fmt.Println("\033[2m------------------------------------------------------------")
	fmt.Printf("ID:\t\t%s\n", resp.ID)
	fmt.Printf("Object:\t\t%s\n", resp.Object)
	fmt.Printf("Created:\t%d\n", resp.Created)
	fmt.Printf("Model:\t\t%s\n", resp.Model)
	fmt.Printf("Usage:\t\tprompt tokens: %d; completion tokens: %d; total tokens: %d\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	fmt.Printf("FinishReason:\t%s\n", finishReason)
	fmt.Printf("Reasoning:\t%s\n", choice.Message.ReasoningContent)
	fmt.Printf("------------------------------------------------------------\033[0m\n\n")
	fmt.Printf("Response:\t%s\n\n", choice.Message.Content)
}
