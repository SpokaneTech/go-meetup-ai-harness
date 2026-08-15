package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/signal"
	"strings"

	"github.com/SpokaneText/go-meetup-ai-harness/internal/api"
	"github.com/SpokaneText/go-meetup-ai-harness/internal/input"
	"golang.org/x/sync/errgroup"
)

const (
	maxIterations = 10
	listToolName  = "list"
	readToolName  = "read"
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
			messages, err = runAgentLoop(ctx, &client, messages)
			if err != nil {
				return fmt.Errorf("run agent loop: %w", err)
			}
			fmt.Print("> ")
		}
		return nil
	})
	return wg.Wait()
}

func runAgentLoop(ctx context.Context, client *api.Client, messages []api.Message) ([]api.Message, error) {
	for range maxIterations {
		resp, err := client.ChatCompletion(ctx, api.ChatCompletionRequest{
			Model:    "deepseek-v4-flash",
			Messages: messages,
			Tools:    toolDefinitions(),
		})
		if err != nil {
			return nil, fmt.Errorf("chat completion error: %w", err)
		}
		if err != nil {
			return nil, fmt.Errorf("chat completion error: %w", err)
		}
		if len(resp.Choices) == 0 {
			fmt.Println("expected at least one choice, got 0")
			continue
		}
		choice := resp.Choices[0]
		printResponse(resp, choice)
		messages = append(messages, api.Message{
			Role:      api.RoleAssistant,
			Content:   choice.Message.Content,
			ToolCalls: choice.Message.ToolCalls,
		})
		if choice.Message.ToolCalls != nil {
			for _, toolCall := range choice.Message.ToolCalls {
				messages = append(messages,
					runTool(ctx, toolCall.ID, toolCall.Function.Name, toolCall.Function.Arguments),
				)
			}
		}
		if choice.FinishReason == api.FinishStop || choice.FinishReason == api.FinishLength {
			break
		}
	}
	return messages, nil
}

func toolDefinitions() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        listToolName,
				Description: "List all files and directories in the requested directory",
				Parameters: []byte(`{
					"type": "object",
					"properties": {
						"path": {
							"type": "string",
							"description": "The path of the directory to list"
						}
					},
					"required": ["path"]
				}`),
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        readToolName,
				Description: "Read the contents of a file",
				Parameters: []byte(`{
					"type": "object",
					"properties": {
						"path": {
							"type": "string",
							"description": "The path of file to read"
						}
					},
					"required": ["path"]
				}`),
			},
		},
	}
}

// runTool calls the appropriate tool. It intentionally never returns an error. Errors should be returned
// to the agent so the agent can potentially fix the tool call.
func runTool(ctx context.Context, toolCallID string, toolName string, toolArgs string) api.Message {
	fmt.Printf("\033[36mrunning tool:\t%s, with args: %s\033[0m\n\n", toolName, toolArgs)
	var content strings.Builder
	switch toolName {
	case listToolName:
		var args struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(toolArgs), &args); err != nil {
			fmt.Fprintf(&content, "invalid arguments: %s", err)
			break
		}
		if args.Path == "" {
			fmt.Fprintf(&content, "path is required")
			break
		}
		files, err := os.ReadDir(args.Path)
		if err != nil {
			fmt.Fprintf(&content, "failed to read directory: %s", err)
			break
		}
		for _, f := range files {
			ftype := "file"
			if f.IsDir() {
				ftype = "directory"
			}
			fmt.Fprintf(&content, "[%s] %s\n", ftype, f.Name())
		}
	case readToolName:
		var args struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(toolArgs), &args); err != nil {
			fmt.Fprintf(&content, "invalid arguments: %s", err)
			break
		}
		if args.Path == "" {
			fmt.Fprintf(&content, "path is required")
			break
		}
		fileContent, err := os.ReadFile(args.Path)
		if err != nil {
			fmt.Fprintf(&content, "failed to read file: %s", err)
			break
		}
		content.WriteString(string(fileContent))
	default:
		fmt.Fprintf(&content, "unknown tool: %s", toolName)
	}
	fmt.Printf("\033[36mreturned:\n---\n%s\n---\033[0m\n\n", content.String()[0:min(len(content.String()), 100)])
	return api.Message{
		Role:       api.RoleTool,
		ToolCallID: toolCallID,
		Content:    content.String(),
	}
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
	fmt.Printf("ToolCalls:\t%v\n", choice.Message.ToolCalls)
	fmt.Printf("Reasoning:\t%s\n", choice.Message.ReasoningContent)
	fmt.Printf("------------------------------------------------------------\033[0m\n\n")
	fmt.Printf("Response:\t%s\n\n", choice.Message.Content)
}
