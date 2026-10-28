import type { InlineExtension } from "../core/extensions/types.ts";
import codemodeExtension from "./codemode/index.ts";
import llamaExtension from "./llama/index.ts";
import mcpExtension from "./mcp/index.ts";
import toolSearchExtension from "./tool-search/index.ts";

// 内置的ExtensionFactory数组
export const builtInExtensions: InlineExtension[] = [
	// llama的扩展
	{ name: "llama.cpp", factory: llamaExtension, builtin: true },
	// Replaceable: an extension that registers `codemode`, `tool_search`, or `/mcp` (such as a third-party
	// MCP extension) takes over instead of running alongside the built-in one.
	// 注册一个codemode的tool
	{ name: "codemode", factory: codemodeExtension, replaceable: true, builtin: true },
	// 注册一个tool-search的tool
	{ name: "tool-search", factory: toolSearchExtension, replaceable: true, builtin: true },
	// 执行mcp相关的拓展
	{ name: "mcp", factory: mcpExtension, replaceable: true, builtin: true },
];
