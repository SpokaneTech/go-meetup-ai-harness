package api

import "encoding/json"

// Role is the role of a message participant.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishFunctionCall  FinishReason = "function_call" // deprecated, pre tool_calls
	FinishContentFilter FinishReason = "content_filter"
)

// ChatCompletionResponse is the top-level response returned from
// POST /v1/chat/completions when stream=false.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice is a single generated completion candidate.
type Choice struct {
	Index        int          `json:"index"`
	Message      Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
}

// Message is a single chat message, used both in requests and in the
// "message" field of a non-streaming response choice.
type Message struct {
	Role             Role       `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Name             string     `json:"name,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"` // set on role="tool" messages
}

// ToolCall represents a single tool/function invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // currently always "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the function name and raw JSON-encoded arguments.
// Arguments is a string (not a json.RawMessage object) per the OpenAI spec,
// so callers must json.Unmarshal it into their own parameter struct.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Usage reports token accounting for the request.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails breaks down prompt token usage (e.g. cached tokens).
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

// CompletionTokensDetails breaks down completion token usage.
type CompletionTokensDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// ---------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------

// ChatCompletionRequest is the body for POST /v1/chat/completions with
// stream=false (or omitted). Pointer types are used for fields where
// nil vs. zero-value matters (the provider applies its own default when
// the field is omitted entirely).
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

// Tool describes a callable function made available to the model.
type Tool struct {
	Type     string       `json:"type"` // currently always "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function schema portion of a Tool.
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object. json.RawMessage keeps this
	// package free of a full JSON Schema type; build/marshal it separately.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// ToolChoice controls whether/which tool the model must call. Marshal as
// the bare string "none" / "auto" / "required", or as a
// ToolChoiceFunction object to force a specific function. Set exactly one
// of the two fields before marshaling; use the helper constructors below.
type ToolChoice struct {
	mode     string // "none", "auto", "required", or "" for a forced function
	function *ToolChoiceFunction
}

// ToolChoiceFunction forces the model to call a specific named function.
type ToolChoiceFunction struct {
	Type     string                 `json:"type"` // "function"
	Function ToolChoiceFunctionName `json:"function"`
}

type ToolChoiceFunctionName struct {
	Name string `json:"name"`
}

func ToolChoiceAuto() ToolChoice     { return ToolChoice{mode: "auto"} }
func ToolChoiceNone() ToolChoice     { return ToolChoice{mode: "none"} }
func ToolChoiceRequired() ToolChoice { return ToolChoice{mode: "required"} }
func ToolChoiceForce(functionName string) ToolChoice {
	return ToolChoice{function: &ToolChoiceFunction{
		Type:     "function",
		Function: ToolChoiceFunctionName{Name: functionName},
	}}
}

func (t ToolChoice) MarshalJSON() ([]byte, error) {
	if t.function != nil {
		return json.Marshal(t.function)
	}
	if t.mode == "" {
		return json.Marshal("auto")
	}
	return json.Marshal(t.mode)
}

func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		t.mode = s
		t.function = nil
		return nil
	}
	var f ToolChoiceFunction
	if err := json.Unmarshal(data, &f); err != nil {
		return err
	}
	t.function = &f
	t.mode = ""
	return nil
}

// ResponseFormat requests plain text, generic JSON, or a JSON Schema-
// constrained response ("structured outputs").
type ResponseFormat struct {
	Type       string          `json:"type"` // "text", "json_object", or "json_schema"
	JSONSchema *JSONSchemaSpec `json:"json_schema,omitempty"`
}

// JSONSchemaSpec is the schema payload when ResponseFormat.Type is "json_schema".
type JSONSchemaSpec struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict *bool           `json:"strict,omitempty"`
}

// ---------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------

// ErrorResponse is the shape returned on non-2xx responses by OpenAI and
// most compatible providers.
type ErrorResponse struct {
	Error APIError `json:"error"`
}

type APIError struct {
	Message string          `json:"message"`
	Type    string          `json:"type,omitempty"`
	Param   *string         `json:"param,omitempty"`
	Code    json.RawMessage `json:"code,omitempty"` // some providers send string, others int/null
}

func (e *APIError) ErrorString() string {
	return e.Message
}
