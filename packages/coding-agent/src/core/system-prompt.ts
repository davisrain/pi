/**
 * System prompt construction and project context loading
 */

import { getSystemMessageText } from "@earendil-works/pi-ai";
import { getDocsPath, getExamplesPath, getReadmePath } from "../config.ts";
import { formatSkillsForPrompt, type Skill } from "./skills.ts";

export interface BuildSystemPromptOptions {
	/** Custom system prompt (replaces the default prefix). */
	// 自定义开头，设置后会替换默认的preamble，同时去掉默认的tools、rules、docs三个section
	customPrompt?: string;
	/** Exact full prompt replacement set by a before_agent_start handler. */
	// 用这段文本完整替换整个prompt，不再分section
	forceSystemPrompt?: string;
	/** Tools to include in prompt. Default: [read, bash, edit, write]. */
	// 当前激活的工具名
	selectedTools?: string[];
	/** Optional one-line tool snippets keyed by tool name. */
	toolSnippets?: Record<string, string>;
	/** Guideline bullets contributed by each tool, keyed by tool name. */
	toolGuidelines?: Record<string, string[]>;
	/** Additional guideline bullets appended to the default system prompt rules. */
	promptGuidelines?: string[];
	/** Text appended from user configuration before project context, skills, and cwd. */
	appendSystemPrompt?: string;
	/** Additional XML-wrapped prompt sections keyed by tag name. */
	sections?: Record<string, string>;
	/** Working directory. */
	cwd: string;
	/** Pre-loaded context files. */
	contextFiles?: Array<{ path: string; content: string }>;
	/** Pre-loaded skills. */
	skills?: Skill[];
}

export type NormalizedBuildSystemPromptOptions = BuildSystemPromptOptions & {
	// 当前激活的工具名
	selectedTools: string[];
	// 写进工具列表里的那一行说明
	toolSnippets: Record<string, string>;
	// 每个工具附带的规则，只有工具被选中时才生效
	toolGuidelines: Record<string, string[]>;
	// 不属于任何工具的额外规则
	promptGuidelines: string[];
	// 用户配置追加的文本
	appendSystemPrompt: string;
	// 扩展自定义的额外section
	sections: Record<string, string>;
	// Agents.md这类项目说明文件
	contextFiles: Array<{ path: string; content: string }>;
	// 已加载的skill列表
	skills: Skill[];
	// 拼出来的大概样子是：
	// You are an expert coding assistant operating inside pi, ...        ← preamble（无标签）
	// <tools>
	// - read: Read file contents
	// - bash: Execute bash commands
	// ...
	// </tools>
	// <rules>
	// - Use bash for file operations like ls, rg, find
	// - Be concise in your responses
	// ...
	// </rules>
	// <docs>...</docs>
	// <project_context>
	// Project-specific instructions and guidelines:
	// <project_instructions path="/Users/.../AGENTS.md">...</project_instructions>
	// </project_context>
	// <skills>...</skills>
	// <cwd>
	// /Users/zhengyudai/code/pi
	// </cwd>
};

/**
 * Ordered system prompt sections, keyed by name. `preamble` is untagged text; every other
 * section is wrapped in a tag of the same name so the model can match later updates to it.
 * These become `SystemMessage.sections` in the transcript.
 */
export type SystemPromptSections = Record<string, string>;

const SYSTEM_PROMPT_SECTION_NAME = /^[a-z][a-z0-9_-]*$/;
/** Normalize prompt input into the mutable, collection-complete shape exposed to extensions. */
export function normalizeBuildSystemPromptOptions(input: BuildSystemPromptOptions): NormalizedBuildSystemPromptOptions {
	return {
		customPrompt: input.customPrompt,
		forceSystemPrompt: input.forceSystemPrompt,
		selectedTools: [...(input.selectedTools ?? ["read", "bash", "edit", "write"])],
		toolSnippets: { ...(input.toolSnippets ?? {}) },
		toolGuidelines: Object.fromEntries(
			Object.entries(input.toolGuidelines ?? {}).map(([name, guidelines]) => [name, [...guidelines]]),
		),
		promptGuidelines: [...(input.promptGuidelines ?? [])],
		appendSystemPrompt: input.appendSystemPrompt ?? "",
		sections: { ...(input.sections ?? {}) },
		cwd: input.cwd,
		contextFiles: (input.contextFiles ?? []).map((file) => ({ ...file })),
		skills: (input.skills ?? []).map((skill) => ({ ...skill })),
	};
}

function renderProjectContext(contextFiles: Array<{ path: string; content: string }>): string {
	return [
		"Project-specific instructions and guidelines:",
		...contextFiles.map(
			({ path, content }) => `<project_instructions path="${path}">\n${content}\n</project_instructions>`,
		),
	].join("\n\n");
}

function buildRules(
	selectedTools: string[],
	toolGuidelines: Record<string, string[]>,
	promptGuidelines: string[],
): string {
	const rules: string[] = [];
	const seen = new Set<string>();
	// 添加rules的函数
	const addRule = (rule: string): void => {
		const normalized = rule.trim();
		if (!normalized || seen.has(normalized)) return;
		seen.add(normalized);
		rules.push(normalized);
	};

	// 判断selectedTools是否包含以下工具：bash powershell grep find ls
	const hasBash = selectedTools.includes("bash");
	const hasPowerShell = selectedTools.includes("powershell");
	const hasGrep = selectedTools.includes("grep");
	const hasFind = selectedTools.includes("find");
	const hasLs = selectedTools.includes("ls");

	// 如果包含bash 或者 powershell 并且 不包含grep find ls
	if ((hasBash || hasPowerShell) && !hasGrep && !hasFind && !hasLs) {
		// 添加使用bash或者powershell来操作文件的规则
		if (hasBash && hasPowerShell) {
			addRule("Use bash or PowerShell for file operations like listing, searching, and finding files");
		} else if (hasPowerShell) {
			addRule("Use PowerShell for file operations like listing, searching, and finding files");
		} else {
			addRule("Use bash for file operations like ls, rg, find");
		}
	}

	// 如果对应的工具有guideline的话，也添加到规则里面
	for (const name of selectedTools) {
		for (const rule of toolGuidelines[name] ?? []) addRule(rule);
	}
	// 将prompt的规则也添加进来
	for (const rule of promptGuidelines) addRule(rule);
	// 添加两条内置的规则
	addRule("Be concise in your responses");
	addRule("Show file paths clearly when working with files");
	// 整合成规则string返回
	return rules.map((rule) => `- ${rule}`).join("\n");
}

/** Build the ordered, independently replaceable sections of the structured system prompt. */
export function buildSystemPromptSections(input: BuildSystemPromptOptions): SystemPromptSections {
	const options = normalizeBuildSystemPromptOptions(input);
	const {
		customPrompt,
		selectedTools,
		toolSnippets,
		toolGuidelines,
		promptGuidelines,
		appendSystemPrompt,
		sections: customSections,
		cwd,
		contextFiles,
		skills,
	} = options;

	// 判断自定义的sections，name如果不符合规范，或者叫preamble，直接报错
	for (const name of Object.keys(customSections)) {
		if (!SYSTEM_PROMPT_SECTION_NAME.test(name) || name === "preamble") {
			throw new Error(`Invalid system prompt section name: ${name}`);
		}
	}


	const promptSections: Record<string, string> = {};
	// 如果存在customPrompt的话
	if (customPrompt) {
		// 将preamble设置为customPrompt
		promptSections.preamble = customPrompt;
	} else {
		// 否则，preamble为指定的下述内容
		promptSections.preamble =
			"You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";
		// 过滤掉那些没有toolSnippet的tool
		const visibleTools = selectedTools.filter((name) => !!toolSnippets[name]);
		// 将可见的tools拼接成 - name: toolSnippet的格式
		const tools =
			visibleTools.length > 0 ? visibleTools.map((name) => `- ${name}: ${toolSnippets[name]}`).join("\n") : "(none)";
		// 添加tools section的内容
		promptSections.tools = `${tools}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.`;
		// 构建rules section的内容
		promptSections.rules = buildRules(selectedTools, toolGuidelines, promptGuidelines);
		// 构建docs section的内容
		promptSections.docs = `Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: ${getReadmePath()}
- Additional docs: ${getDocsPath()}
- Examples: ${getExamplesPath()} (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)`;
	}

	// 如果存在附加的systemPrompt，那么添加附录 section
	if (appendSystemPrompt) promptSections.addendum = appendSystemPrompt;
	// 如果上下文文件存在的话，添加project_context section
	if (contextFiles.length > 0) promptSections.project_context = renderProjectContext(contextFiles);
	// 如果工具中包含read或者bash的话，说明是有读取skill文件的工具的
	const skillFileReadTool = (["read", "bash"] as const).find((tool) => selectedTools.includes(tool));
	// 那么就添加skills section
	if (skillFileReadTool && skills.length > 0) {
		// 格式化skills的信息用于prompt
		const skillsPrompt = formatSkillsForPrompt(skills, skillFileReadTool).trim();
		if (skillsPrompt) promptSections.skills = skillsPrompt;
	}
	// 添加cwd section，将\替换成/
	promptSections.cwd = cwd.replace(/\\/g, "/");
	// 将自定义的section添加进去
	for (const [name, content] of Object.entries(customSections)) {
		if (content) promptSections[name] = content;
	}

	// 先将preamble放进sections中，这种类型的section不需要标签
	const sections: SystemPromptSections = { preamble: promptSections.preamble };
	// 然后将其他section都加上标签放入sections中
	for (const [name, content] of Object.entries(promptSections)) {
		if (name !== "preamble") sections[name] = `<${name}>\n${content}\n</${name}>`;
	}
	// 返回sections
	return sections;
}

/**
 * The complete prompt state for `input`. A forced prompt is opaque and lives in `content`
 * with no sections; otherwise `content` is empty and the structured sections carry the prompt.
 */
export function buildSystemPromptState(input: BuildSystemPromptOptions): {
	content: string;
	sections?: SystemPromptSections;
} {
	if (input.forceSystemPrompt !== undefined) return { content: input.forceSystemPrompt };
	return { content: "", sections: buildSystemPromptSections(input) };
}

/** Build the system prompt text, rendered exactly as the transcript's system message replays it. */
export function buildSystemPrompt(input: BuildSystemPromptOptions): string {
	return getSystemMessageText({ role: "system", ...buildSystemPromptState(input), timestamp: 0 });
}

/**
 * Diff the sections the model currently has (replayed from the transcript, so never null)
 * against the desired ones. Returns a `SystemMessage.sections` patch, or undefined when
 * nothing changed.
 */
export function diffSystemPromptSections(
	previous: Record<string, string | null>,
	current: SystemPromptSections,
): Record<string, string | null> | undefined {
	const patch: Record<string, string | null> = {};
	// 遍历当前的sections
	for (const [name, text] of Object.entries(current)) {
		// 如果section的内容和以前的不相等的话，放入到patch中
		if (previous[name] !== text) patch[name] = text;
	}
	// 然后遍历以前的sections
	for (const name of Object.keys(previous)) {
		// 如果在当前的sections中没有找到的话，放入到patch中，且value为null
		if (current[name] === undefined) patch[name] = null;
	}
	// 如果patch中存在元素，返回
	return Object.keys(patch).length > 0 ? patch : undefined;
}
