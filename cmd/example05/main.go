package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/SpokaneText/go-meetup-ai-harness/internal/api"
	"github.com/SpokaneText/go-meetup-ai-harness/internal/input"
	"golang.org/x/sync/errgroup"
)

const (
	maxIterations = 10
	listToolName  = "list"
	readToolName  = "read"
	editToolName  = "edit"
	bashToolName  = "bash"
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
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: editToolName,
				Description: `Edit the contents of a file or create new ones. In order to edit a file,
				provide the exact string you want to replace and the string you want to replace it with.
				The old content must match exactly one occurance in the file or it will be blocked.
				If you want to create a new file then provide an empty string for the old_content. If the
				file exists then the file create will be blocked. Sub-directories will be created
				automaically if they do not exist.`,
				Parameters: []byte(`{
					"type": "object",
					"properties": {
						"path": {
							"type": "string",
							"description": "The path of file to edit"
						},
						"old_content": {
							"type": "string",
							"description": "The old content of the file to be replaced"
						},
						"new_content": {
							"type": "string",
							"description": "The new content that the old content will be replaced with"
						}
					},
					"required": ["path"]
				}`),
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        bashToolName,
				Description: "Execute a shell command with bash",
				Parameters: []byte(`{
					"type": "object",
					"properties": {
						"command": {
							"type": "string",
							"description": "The command to execute"
						}
					},
					"required": ["command"]
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
	case editToolName:
		var args struct {
			Path       string `json:"path"`
			OldContent string `json:"old_content"`
			NewContent string `json:"new_content"`
		}
		if err := json.Unmarshal([]byte(toolArgs), &args); err != nil {
			fmt.Fprintf(&content, "invalid arguments: %s", err)
			break
		}
		hasError := false
		if args.Path == "" {
			fmt.Fprintf(&content, "path is required")
			hasError = true
		}
		if args.NewContent == "" {
			fmt.Fprintf(&content, "new_content is required")
			hasError = true
		}
		if hasError {
			break
		}
		err := editFile(args.Path, args.OldContent, args.NewContent)
		if err != nil {
			fmt.Fprintf(&content, "failed to edit file: %s", err)
			break
		}
		fmt.Fprintf(&content, "successfully edited file %s", args.Path)
	case bashToolName:
		var args struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal([]byte(toolArgs), &args); err != nil {
			fmt.Fprintf(&content, "invalid arguments: %s", err)
			break
		}
		if args.Command == "" {
			content.WriteString("command is required but was blank")
			break
		}
		var stdout, stderr bytes.Buffer
		cmd := exec.CommandContext(ctx, "bash", "-c", args.Command) //nolint:gosec // allowing arbitrary command execution is the point
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		exitCode := 0
		var exitError *exec.ExitError
		switch {
		case err == nil:
			// success
		case errors.As(err, &exitError):
			// The command ran but exited non-zero; report the exit code
			// along with stdout/stderr so the agent can see what happened.
			exitCode = exitError.ExitCode()
		default:
			fmt.Fprintf(&content, "command failed to start [%s]: %s", args.Command, err)
		}
		fmt.Fprintf(&content,
			"exit code: %d\n--- stdout ---\n%s\n--- stderr ---\n%s",
			exitCode,
			stdout.String(),
			stderr.String(),
		)
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

func editFile(path, oldContent, newContent string) error {
	// A few use cases to verify.
	// 1. If old content is blank then we are creating a new file.
	//    Make sure that there is no file at the requested path.
	if oldContent == "" {
		return createFile(path, newContent)
	}

	fileContents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading file to compare contents: %s", err)
	}

	// 2. If old content is populated then a file must exist at the
	//    path and exist exactly one instance of the oldContent.
	if count := strings.Count(string(fileContents), oldContent); count != 1 {
		return fmt.Errorf("found %d instances of old_content. Be more specific with your old content to make sure that only one match is found", count)
	}

	newFileContents := strings.Replace(string(fileContents), oldContent, newContent, 1)
	return os.WriteFile(path, []byte(newFileContents), 0o644)
}

func createFile(path, newContent string) error {
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		// File exists so we need to let the agent know.
		return errors.New(
			`tool 'edit' cannot create '{file_path}': file already
				exists. To edit it, provide a non-empty, unique
				` + "`old_string`" + ` that matches the current contents
				exactly.
			`)
	}

	// Create necessary subdirectories to the file before attempting to write it
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating parent directories: %s", err)
	}

	// Then write the contents
	if err := os.WriteFile(path, []byte(newContent), 0o664); err != nil {
		return fmt.Errorf("unable to create new file: %s", err)
	}

	return nil
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
