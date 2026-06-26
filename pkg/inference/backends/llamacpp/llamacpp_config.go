package llamacpp

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"

	"github.com/docker/model-runner/pkg/distribution/types"
	"github.com/docker/model-runner/pkg/inference"
	"github.com/docker/model-runner/pkg/logging"
)

const UnlimitedContextSize = -1

// Config is the configuration for the llama.cpp backend.
type Config struct {
	// Args are the base arguments that are always included.
	Args []string
	// Logger is used for warning when user flags replace defaults.
	Logger logging.Logger
}

// NewDefaultLlamaCppConfig creates a new LlamaCppConfig with default values.
func NewDefaultLlamaCppConfig() *Config {
	return NewDefaultLlamaCppConfigWithLogger(nil)
}

// NewDefaultLlamaCppConfigWithLogger creates a new LlamaCppConfig with default values and a logger
// for emitting warnings when user-specified flags replace defaults.
func NewDefaultLlamaCppConfigWithLogger(logger logging.Logger) *Config {
	args := []string{"-ngl", "999", "--metrics"}

	// Special case for macOS (Apple), optimization
	if runtime.GOOS == "darwin" {
		args = append(args, "--no-mmap")
	}

	// Special case for ARM64
	if runtime.GOARCH == "arm64" {
		// Using a thread count equal to core count results in bad performance, and there seems to be little to no gain
		// in going beyond core_count/2.
		if !containsArg(args, "--threads") {
			nThreads := max(2, runtime.NumCPU()/2)
			args = append(args, "--threads", strconv.Itoa(nThreads))
		}
	}

	return &Config{
		Args:   args,
		Logger: logger,
	}
}

// GetArgs implements BackendConfig.GetArgs.
func (c *Config) GetArgs(bundle types.ModelBundle, socket string, mode inference.BackendMode, config *inference.BackendConfiguration) ([]string, error) {
	// Start with the arguments from LlamaCppConfig, replacing any defaults
	// that conflict with user-specified runtime flags.
	var userFlags []string
	if config != nil {
		userFlags = config.RuntimeFlags
	}
	args := replaceConflictingDefaults(c.Args, userFlags, c.Logger)

	modelPath := bundle.GGUFPath()
	if modelPath == "" {
		return nil, fmt.Errorf("GGUF file required by llama.cpp backend")
	}

	// Add model and socket arguments
	args = append(args, "--model", modelPath, "--host", socket)

	// Add mode-specific arguments
	switch mode {
	case inference.BackendModeCompletion:
		// Add arguments for chat template file
		if path := bundle.ChatTemplatePath(); path != "" {
			args = append(args, "--chat-template-file", path)
		}
	case inference.BackendModeEmbedding:
		args = append(args, "--embeddings")
	case inference.BackendModeReranking:
		args = append(args, "--embeddings", "--reranking")
	case inference.BackendModeImageGeneration:
		return nil, fmt.Errorf("unsupported backend mode %q", mode)
	}

	if budget := GetReasoningBudget(config); budget != nil {
		args = append(args, "--reasoning-budget", strconv.FormatInt(int64(*budget), 10))
	}

	// Add context size from model config or backend config
	contextSize := GetContextSize(bundle.RuntimeConfig(), config)
	if contextSize != nil {
		args = append(args, "--ctx-size", strconv.FormatInt(int64(*contextSize), 10))
	}

	// Append user-specified runtime flags (conflicting defaults already removed)
	args = append(args, userFlags...)

	// Add arguments for Multimodal projector or jinja (they are mutually exclusive)
	if path := bundle.MMPROJPath(); path != "" {
		args = append(args, "--mmproj", path)
	} else {
		args = append(args, "--jinja")
	}

	return args, nil
}

func GetContextSize(modelCfg types.ModelConfig, backendCfg *inference.BackendConfiguration) *int32 {
	// Backend config takes precedence (runtime configuration via docker model configure / Ollama API num_ctx)
	if backendCfg != nil && backendCfg.ContextSize != nil && (*backendCfg.ContextSize == UnlimitedContextSize || *backendCfg.ContextSize > 0) {
		return backendCfg.ContextSize
	}
	// Fallback to model config (set at packaging time via docker model package --context-size)
	if modelCfg != nil {
		if ctxSize := modelCfg.GetContextSize(); ctxSize != nil && (*ctxSize == UnlimitedContextSize || *ctxSize > 0) {
			return ctxSize
		}
	}
	return nil
}

func GetReasoningBudget(backendCfg *inference.BackendConfiguration) *int32 {
	if backendCfg != nil && backendCfg.LlamaCpp != nil && backendCfg.LlamaCpp.ReasoningBudget != nil {
		return backendCfg.LlamaCpp.ReasoningBudget
	}
	return nil
}

// containsArg checks if the given argument is already in the args slice.
func containsArg(args []string, arg string) bool {
	for _, a := range args {
		if a == arg {
			return true
		}
	}
	return false
}

// flagAliases maps canonical flag names to all their aliases.
// When a user specifies any alias, it replaces the default for that canonical flag.
var flagAliases = map[string][]string{
	"-ngl":         {"-ngl", "--gpu-layers", "--n-gpu-layers"},
	"--metrics":    {"--metrics"},
	"--no-mmap":    {"--no-mmap", "--mmap"},
	"--threads":    {"--threads", "-t"},
}

// getCanonicalFlag returns the canonical flag name for a given flag, or empty string if unknown.
func getCanonicalFlag(flag string) string {
	for canonical, aliases := range flagAliases {
		for _, alias := range aliases {
			if flag == alias {
				return canonical
			}
		}
	}
	return ""
}

// parseUserFlags extracts a map of canonical flag names to their user-provided values.
// It handles both space-separated format ("--flag", "value") and equals format ("--flag=value").
func parseUserFlags(flags []string) map[string]string {
	result := make(map[string]string)
	for i, flag := range flags {
		if !strings.HasPrefix(flag, "-") {
			continue // skip values
		}
		// Handle --flag=value format
		if eqIdx := strings.Index(flag, "="); eqIdx > 0 {
			flagKey := flag[:eqIdx]
			value := flag[eqIdx+1:]
			if canonical := getCanonicalFlag(flagKey); canonical != "" {
				result[canonical] = value
			}
			continue
		}
		// Handle --flag value format
		if canonical := getCanonicalFlag(flag); canonical != "" {
			if i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "-") {
				// Next element is a value
				result[canonical] = flags[i+1]
			} else {
				// Boolean flag with no value (e.g., --no-metrics, --no-jinja)
				result[canonical] = ""
			}
		}
	}
	return result
}

// replaceConflictingDefaults returns a copy of defaults with conflicting flags removed,
// and logs a warning for each replacement.
func replaceConflictingDefaults(defaults, userFlags []string, logger logging.Logger) []string {
	userFlagsMap := parseUserFlags(userFlags)
	if len(userFlagsMap) == 0 {
		return append([]string{}, defaults...)
	}

	// Build a set of default flag keys (only flags, not values) mapped to canonical names.
	// Also track positions so we can remove flag+value pairs.
	type defaultInfo struct {
		canonical string
		idx       int // index of the flag itself
	}
	defaultFlags := make(map[string]defaultInfo)
	for i, arg := range defaults {
		if !strings.HasPrefix(arg, "-") {
			continue // skip values
		}
		if canonical := getCanonicalFlag(arg); canonical != "" {
			defaultFlags[canonical] = defaultInfo{canonical: canonical, idx: i}
		}
	}

	// Find which defaults conflict with user flags.
	var conflicts []string
	for canonical := range userFlagsMap {
		if _, ok := defaultFlags[canonical]; ok {
			conflicts = append(conflicts, canonical)
		}
	}

	// Log warnings for each conflict.
	for _, canonical := range conflicts {
		userValue := userFlagsMap[canonical]
		if logger != nil {
			logger.Warn(
				"User-specified flag replaces default; please specify only one to avoid duplicate flag warnings",
				"flag", canonical,
				"user_value", userValue,
			)
		}
	}

	// Build the result: remove conflicting defaults, keep the rest.
	// We need to identify which indices to skip (both the flag and its value).
	skipIdx := make(map[int]bool)
	for _, canonical := range conflicts {
		info := defaultFlags[canonical]
		skipIdx[info.idx] = true
		// Also skip the value (next index) if it exists and is not itself a flag.
		if info.idx+1 < len(defaults) && !strings.HasPrefix(defaults[info.idx+1], "-") {
			skipIdx[info.idx+1] = true
		}
	}

	result := make([]string, 0, len(defaults))
	for i, arg := range defaults {
		if !skipIdx[i] {
			result = append(result, arg)
		}
	}
	return result
}
