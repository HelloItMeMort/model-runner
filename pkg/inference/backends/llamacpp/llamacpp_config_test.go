package llamacpp

import (
	"context"
	"log/slog"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/docker/model-runner/pkg/distribution/types"
	"github.com/docker/model-runner/pkg/inference"
)

func TestNewDefaultLlamaCppConfig(t *testing.T) {
	config := NewDefaultLlamaCppConfig()

	// Test that --jinja is NOT in default args (it will be added conditionally in GetArgs)
	if containsArg(config.Args, "--jinja") {
		t.Error("Did not expect --jinja argument in default config (it should be added conditionally)")
	}

	// Test -ngl argument and its value
	nglIndex := -1
	for i, arg := range config.Args {
		if arg == "-ngl" {
			nglIndex = i
			break
		}
	}
	if nglIndex == -1 {
		t.Error("Expected -ngl argument to be present")
	}
	if nglIndex+1 >= len(config.Args) {
		t.Error("No value found after -ngl argument")
	}
	if config.Args[nglIndex+1] != "999" {
		t.Errorf("Expected -ngl value to be 999, got %s", config.Args[nglIndex+1])
	}

	// Test macOS (Apple) specific case
	if runtime.GOOS == "darwin" {
		if !containsArg(config.Args, "--no-mmap") {
			t.Error("Expected --no-mmap argument to be present on macOS")
		}
	} else if containsArg(config.Args, "--no-mmap") {
		// On non-macOS systems, --no-mmap should not be present by default
		t.Error("Did not expect --no-mmap argument to be present on non-macOS systems")
	}

	// Test Windows ARM64 specific case
	if runtime.GOOS == "windows" && runtime.GOARCH == "arm64" {
		if !containsArg(config.Args, "--threads") {
			t.Error("Expected --threads argument to be present on Windows ARM64")
		}
		threadsIndex := -1
		for i, arg := range config.Args {
			if arg == "--threads" {
				threadsIndex = i
				break
			}
		}
		if threadsIndex == -1 {
			t.Error("Could not find --threads argument")
		}
		if threadsIndex+1 >= len(config.Args) {
			t.Error("No value found after --threads argument")
		}
		threads, err := strconv.Atoi(config.Args[threadsIndex+1])
		if err != nil {
			t.Errorf("Failed to parse thread count: %v", err)
		}
		if threads > runtime.NumCPU()/2 {
			t.Errorf("Thread count %d exceeds maximum allowed value of %d", threads, runtime.NumCPU()/2)
		}
		if threads < 1 {
			t.Error("Thread count is less than 1")
		}
	}
}

func TestGetArgs(t *testing.T) {
	config := NewDefaultLlamaCppConfig()
	modelPath := "/path/to/model"
	socket := "unix:///tmp/socket"

	// Build base expected args based on architecture and OS
	baseArgs := []string{"-ngl", "999", "--metrics"}
	if runtime.GOOS == "darwin" {
		baseArgs = append(baseArgs, "--no-mmap")
	}
	if runtime.GOARCH == "arm64" {
		nThreads := max(2, runtime.NumCPU()/2)
		baseArgs = append(baseArgs, "--threads", strconv.Itoa(nThreads))
	}

	tests := []struct {
		name     string
		bundle   types.ModelBundle
		mode     inference.BackendMode
		config   *inference.BackendConfiguration
		expected []string
	}{
		{
			name: "completion mode",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--jinja",
			),
		},
		{
			name: "embedding mode",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--jinja",
			),
		},
		{
			name: "context size from backend config",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(1234),
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--ctx-size", "1234",
				"--jinja",
			),
		},
		{
			name: "unlimited context size from backend config",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(-1),
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--ctx-size", "-1",
				"--jinja",
			),
		},
		{
			name: "0 context size from backend config ignored",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(0),
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--jinja",
			),
		},
		{
			name: "invalid context size from backend config ignored",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(-2),
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--jinja",
			),
		},
		{
			name: "backend config takes precedence over model config",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
				config: &types.Config{
					ContextSize: int32ptr(2096),
				},
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(1234),
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--ctx-size", "1234", // backend config takes precedence
				"--jinja",
			),
		},
		{
			name: "model config used when no backend config",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
				config: &types.Config{
					ContextSize: int32ptr(2096),
				},
			},
			config: nil,
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--ctx-size", "2096", // model config used as fallback
				"--jinja",
			),
		},
		{
			name: "chat template from model artifact",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath:     modelPath,
				templatePath: "/path/to/bundle/template.jinja",
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--chat-template-file", "/path/to/bundle/template.jinja",
				"--jinja",
			),
		},
		{
			name: "raw flags from backend config",
			mode: inference.BackendModeEmbedding,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				RuntimeFlags: []string{"--some", "flag"},
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--embeddings",
				"--some", "flag",
				"--jinja",
			),
		},
		{
			name: "multimodal projector removes jinja",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath:   modelPath,
				mmprojPath: "/path/to/model.mmproj",
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--mmproj", "/path/to/model.mmproj",
			),
		},
		{
			name: "reasoning budget enabled (-1 unlimited)",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				LlamaCpp: &inference.LlamaCppConfig{
					ReasoningBudget: int32ptr(-1),
				},
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--reasoning-budget", "-1",
				"--jinja",
			),
		},
		{
			name: "reasoning budget disabled (0)",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				LlamaCpp: &inference.LlamaCppConfig{
					ReasoningBudget: int32ptr(0),
				},
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--reasoning-budget", "0",
				"--jinja",
			),
		},
		{
			name: "nil LlamaCpp config (no reasoning budget)",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				LlamaCpp: nil,
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--jinja",
			),
		},
		{
			name: "LlamaCpp config with nil reasoning budget",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				LlamaCpp: &inference.LlamaCppConfig{
					ReasoningBudget: nil,
				},
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--jinja",
			),
		},
		{
			name: "reasoning budget with context size",
			mode: inference.BackendModeCompletion,
			bundle: &fakeBundle{
				ggufPath: modelPath,
			},
			config: &inference.BackendConfiguration{
				ContextSize: int32ptr(8192),
				LlamaCpp: &inference.LlamaCppConfig{
					ReasoningBudget: int32ptr(2048),
				},
			},
			expected: append(slices.Clone(baseArgs),
				"--model", modelPath,
				"--host", socket,
				"--reasoning-budget", "2048",
				"--ctx-size", "8192",
				"--jinja",
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, err := config.GetArgs(tt.bundle, socket, tt.mode, tt.config)
			if err != nil {
				t.Errorf("GetArgs() error = %v", err)
			}

			// Check that all expected arguments are present and in the correct order
			expectedIndex := 0
			for i := 0; i < len(args); i++ {
				if expectedIndex >= len(tt.expected) {
					t.Errorf("Unexpected extra argument: %s", args[i])
					continue
				}

				if args[i] != tt.expected[expectedIndex] {
					t.Errorf("Expected argument %s at position %d, got %s", tt.expected[expectedIndex], i, args[i])
					continue
				}

				// If this is a flag that takes a value, check the next argument
				if i+1 < len(args) && (args[i] == "-ngl" || args[i] == "--model" || args[i] == "--host") {
					expectedIndex++
					if args[i+1] != tt.expected[expectedIndex] {
						t.Errorf("Expected value %s for flag %s, got %s", tt.expected[expectedIndex], args[i], args[i+1])
					}
					i++ // Skip the value in the next iteration
				}
				expectedIndex++
			}

			if expectedIndex != len(tt.expected) {
				t.Errorf("Missing expected arguments. Got %d arguments, expected %d", expectedIndex, len(tt.expected))
			}
		})
	}
}

func TestContainsArg(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		arg      string
		expected bool
	}{
		{
			name:     "argument exists",
			args:     []string{"--arg1", "--arg2", "--arg3"},
			arg:      "--arg2",
			expected: true,
		},
		{
			name:     "argument does not exist",
			args:     []string{"--arg1", "--arg2", "--arg3"},
			arg:      "--arg4",
			expected: false,
		},
		{
			name:     "empty args slice",
			args:     []string{},
			arg:      "--arg1",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containsArg(tt.args, tt.arg)
			if result != tt.expected {
				t.Errorf("containsArg(%v, %s) = %v, want %v", tt.args, tt.arg, result, tt.expected)
			}
		})
	}
}

func TestGetCanonicalFlag(t *testing.T) {
	tests := []struct {
		name     string
		flag     string
		expected string
	}{
		{"long flag -ngl", "-ngl", "-ngl"},
		{"long flag --gpu-layers", "--gpu-layers", "-ngl"},
		{"long flag --n-gpu-layers", "--n-gpu-layers", "-ngl"},
		{"long flag --metrics", "--metrics", "--metrics"},
		{"long flag --no-metrics", "--no-metrics", "--metrics"},
		{"long flag --no-mmap", "--no-mmap", "--no-mmap"},
		{"long flag --mmap", "--mmap", "--no-mmap"},
		{"short flag -t", "-t", "--threads"},
		{"long flag --threads", "--threads", "--threads"},
		{"unknown flag", "--unknown", ""},
		{"value not a flag", "999", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getCanonicalFlag(tt.flag)
			if result != tt.expected {
				t.Errorf("getCanonicalFlag(%q) = %q, want %q", tt.flag, result, tt.expected)
			}
		})
	}
}

func TestParseUserFlags(t *testing.T) {
	tests := []struct {
		name     string
		flags    []string
		expected map[string]string
	}{
		{
			name:     "empty flags",
			flags:    []string{},
			expected: map[string]string{},
		},
		{
			name:     "non-flag values ignored",
			flags:    []string{"999", "some-value"},
			expected: map[string]string{},
		},
		{
			name:  "space-separated flag with value",
			flags: []string{"-ngl", "50"},
			expected: map[string]string{
				"-ngl": "50",
			},
		},
		{
			name:  "equals format flag with value",
			flags: []string{"-ngl=50"},
			expected: map[string]string{
				"-ngl": "50",
			},
		},
		{
			name:  "alias --gpu-layers",
			flags: []string{"--gpu-layers", "75"},
			expected: map[string]string{
				"-ngl": "75",
			},
		},
		{
			name:  "alias --n-gpu-layers",
			flags: []string{"--n-gpu-layers", "30"},
			expected: map[string]string{
				"-ngl": "30",
			},
		},
		{
			name:  "multiple flags",
			flags: []string{"-ngl", "50", "--threads", "8"},
			expected: map[string]string{
				"-ngl":      "50",
				"--threads": "8",
			},
		},
		{
			name:  "non-alias flag ignored",
			flags: []string{"--some", "flag"},
			expected: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseUserFlags(tt.flags)
			if len(result) != len(tt.expected) {
				t.Errorf("parseUserFlags(%v) = %v, expected %v", tt.flags, result, tt.expected)
				return
			}
			for k, v := range tt.expected {
				if result[k] != v {
					t.Errorf("parseUserFlags(%v)[%q] = %q, want %q", tt.flags, k, result[k], v)
				}
			}
		})
	}
}

func TestReplaceConflictingDefaults_WithLogger(t *testing.T) {
	// Test that warnings are logged when defaults are replaced.
	var warnedMsg string
	handler := &captureHandler{warnFn: func(msg string, keysAndValues ...any) {
		warnedMsg = msg
	}}
	logger := slog.New(handler)

	defaults := []string{"-ngl", "999", "--metrics"}
	user := []string{"-ngl", "50"}
	result := replaceConflictingDefaults(defaults, user, logger)

	if warnedMsg == "" {
		t.Error("expected warning to be logged when default is replaced")
	}
	if warnedMsg != "" && !strings.Contains(warnedMsg, "User-specified flag replaces default") {
		t.Errorf("expected warning about replacing default, got: %s", warnedMsg)
	}
	if len(result) != 1 || result[0] != "--metrics" {
		t.Errorf("expected only --metrics to remain, got %v", result)
	}
}

type captureHandler struct {
	warnFn func(msg string, keysAndValues ...any)
}

func (h *captureHandler) Enabled(_ context.Context, level slog.Level) bool { return true }
func (h *captureHandler) Handle(_ context.Context, r slog.Record) error {
	if r.Level == slog.LevelWarn && h.warnFn != nil {
		h.warnFn(r.Message)
	}
	return nil
}
func (h *captureHandler) WithAttrs(attrs []slog.Attr) slog.Handler { return h }
func (h *captureHandler) WithGroup(name string) slog.Handler       { return h }

func TestReplaceConflictingDefaults(t *testing.T) {
	tests := []struct {
		name     string
		defaults []string
		user     []string
		expected []string
	}{
		{
			name:     "no user flags, defaults unchanged",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{},
			expected: []string{"-ngl", "999", "--metrics"},
		},
		{
			name:     "no user flags, nil user slice",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     nil,
			expected: []string{"-ngl", "999", "--metrics"},
		},
		{
			name:     "user flag replaces -ngl default (user flags not included, caller appends them)",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"-ngl", "50"},
			expected: []string{"--metrics"},
		},
		{
			name:     "user flag replaces -ngl via alias --gpu-layers",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"--gpu-layers", "75"},
			expected: []string{"--metrics"},
		},
		{
			name:     "user flag replaces -ngl via alias --n-gpu-layers",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"--n-gpu-layers", "30"},
			expected: []string{"--metrics"},
		},
		{
			name:     "user flag replaces -ngl via equals format",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"-ngl=50"},
			expected: []string{"--metrics"},
		},
		{
			name:     "user flag replaces --metrics",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"--no-metrics"},
			expected: []string{"-ngl", "999"},
		},
		{
			name:     "user flag replaces --no-mmap default on macOS",
			defaults: []string{"-ngl", "999", "--metrics", "--no-mmap"},
			user:     []string{"--mmap"},
			expected: []string{"-ngl", "999", "--metrics"},
		},
		{
			name:     "user flag replaces --threads default on arm64",
			defaults: []string{"-ngl", "999", "--metrics", "--threads", "4"},
			user:     []string{"-t", "8"},
			expected: []string{"-ngl", "999", "--metrics"},
		},
		{
			name:     "non-conflicting user flag does not affect defaults",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"--some", "flag"},
			expected: []string{"-ngl", "999", "--metrics"},
		},
		{
			name:     "multiple conflicting and non-conflicting flags",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"-ngl", "50", "--some", "flag"},
			expected: []string{"--metrics"},
		},
		{
			name:     "no conflict, all defaults preserved",
			defaults: []string{"-ngl", "999", "--metrics"},
			user:     []string{"--threads", "4"},
			expected: []string{"-ngl", "999", "--metrics"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := replaceConflictingDefaults(tt.defaults, tt.user, nil)
			if len(result) != len(tt.expected) {
				t.Errorf("replaceConflictingDefaults(%v, %v) = %v (len=%d), expected %v (len=%d)",
					tt.defaults, tt.user, result, len(result), tt.expected, len(tt.expected))
				return
			}
			for i, v := range result {
				if v != tt.expected[i] {
					t.Errorf("replaceConflictingDefaults(%v, %v)[%d] = %q, want %q",
						tt.defaults, tt.user, i, v, tt.expected[i])
				}
			}
		})
	}
}

var _ types.ModelBundle = &fakeBundle{}

type fakeBundle struct {
	ggufPath     string
	config       *types.Config
	templatePath string
	mmprojPath   string
}

func (f *fakeBundle) ChatTemplatePath() string {
	return f.templatePath
}

func (f *fakeBundle) RootDir() string {
	panic("shouldn't be called")
}

func (f *fakeBundle) GGUFPath() string {
	return f.ggufPath
}

func (f *fakeBundle) MMPROJPath() string {
	return f.mmprojPath
}

func (f *fakeBundle) SafetensorsPath() string {
	return ""
}

func (f *fakeBundle) DDUFPath() string {
	return ""
}

func (f *fakeBundle) RuntimeConfig() types.ModelConfig {
	if f.config == nil {
		return nil
	}
	return f.config
}

func int32ptr(n int32) *int32 {
	return &n
}
