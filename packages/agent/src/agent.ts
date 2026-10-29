import {
	createInitialSystemMessage,
	getCurrentSystemMessage,
	getCurrentSystemPrompt,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingBudgets,
	type Transport,
	toToolDeclaration,
} from "@earendil-works/pi-ai";
import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.ts";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	BeforeToolCallContext,
	BeforeToolCallResult,
	FinishTurn,
	PrepareNextTurnContext,
	PrepareRequest,
	QueueMode,
	StreamFn,
	ToolExecutionMode,
} from "./types.ts";

export type { QueueMode } from "./types.ts";

function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) =>
			message.role === "system" ||
			message.role === "user" ||
			message.role === "assistant" ||
			message.role === "toolResult",
	);
}

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const DEFAULT_MODEL = {
	id: "unknown",
	name: "unknown",
	api: "unknown",
	provider: "unknown",
	baseUrl: "",
	reasoning: false,
	input: [],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 0,
	maxTokens: 0,
} satisfies Model<any>;

type MutableAgentState = Omit<AgentState, "isStreaming" | "streamingMessage" | "pendingToolCalls" | "errorMessage"> & {
	isStreaming: boolean;
	streamingMessage?: AgentMessage;
	pendingToolCalls: Set<string>;
	errorMessage?: string;
};

/** Initial state for {@link Agent}. `systemPrompt` and `tools` become the leading system message unless `messages` already starts with one. */
export type AgentInitialState = Partial<
	Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">
>;

function createMutableAgentState(initialState?: AgentInitialState): MutableAgentState {
	let tools = initialState?.tools?.slice() ?? [];
	let messages = initialState?.messages?.slice() ?? [];
	const initialMessage = createInitialSystemMessage(initialState?.systemPrompt, tools.map(toToolDeclaration));
	if (messages[0]?.role !== "system" && initialMessage) messages.unshift(initialMessage);

	return {
		get systemPrompt() {
			return getCurrentSystemPrompt(messages);
		},
		model: initialState?.model ?? DEFAULT_MODEL,
		thinkingLevel: initialState?.thinkingLevel ?? "off",
		get tools() {
			return tools;
		},
		set tools(nextTools: AgentTool<any>[]) {
			tools = nextTools.slice();
		},
		get messages() {
			return messages;
		},
		set messages(nextMessages: AgentMessage[]) {
			messages = nextMessages.slice();
		},
		isStreaming: false,
		streamingMessage: undefined,
		pendingToolCalls: new Set<string>(),
		errorMessage: undefined,
	};
}

/** Options for constructing an {@link Agent}. */
export interface AgentOptions {
	initialState?: AgentInitialState;
	convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	streamFn: StreamFn;
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
	finishTurn?: FinishTurn;
	prepareRequest?: PrepareRequest;
	prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	sessionId?: string;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
	toolExecution?: ToolExecutionMode;
}

class PendingMessageQueue {
	private messages: AgentMessage[] = [];
	public mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	enqueue(message: AgentMessage): void {
		this.messages.push(message);
	}

	hasItems(): boolean {
		return this.messages.length > 0;
	}

	peek(): AgentMessage[] {
		if (this.mode === "all") return this.messages.slice();
		const first = this.messages[0];
		return first ? [first] : [];
	}

	drain(): AgentMessage[] {
		const drained = this.peek();
		this.messages = this.messages.slice(drained.length);
		return drained;
	}

	clear(): void {
		this.messages = [];
	}
}

type ActiveRun = {
	promise: Promise<void>;
	resolve: () => void;
	abortController: AbortController;
};

/**
 * Stateful wrapper around the low-level agent loop.
 *
 * `Agent` owns the current transcript, emits lifecycle events, executes tools,
 * and exposes queueing APIs for steering and follow-up messages.
 */
export class Agent {
	private _state: MutableAgentState;
	private readonly listeners = new Set<(event: AgentEvent, signal: AbortSignal) => Promise<void> | void>();
	// 引导消息队列，agent在每次执行完tool之后检查该队列，如果发现不为空，将剩余的tool丢掉，使用引导消息继续调用llm
	private readonly steeringQueue: PendingMessageQueue;
	// 跟进消息队列，只有当agent发现没有tool执行的时候，才会查看这个队列获取消息
	private readonly followUpQueue: PendingMessageQueue;

	public convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	public transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	public streamFunction: StreamFn;
	public getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	public onPayload?: SimpleStreamOptions["onPayload"];
	public onResponse?: SimpleStreamOptions["onResponse"];
	public onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	public beforeToolCall?: (
		context: BeforeToolCallContext,
		signal?: AbortSignal,
	) => Promise<BeforeToolCallResult | undefined>;
	public afterToolCall?: (
		context: AfterToolCallContext,
		signal?: AbortSignal,
	) => Promise<AfterToolCallResult | undefined>;
	public finishTurn?: FinishTurn;
	public prepareRequest?: PrepareRequest;
	public prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	private activeRun?: ActiveRun;
	/** Session identifier forwarded to providers for cache-aware backends. */
	public sessionId?: string;
	/** Optional per-level thinking token budgets forwarded to the stream function. */
	public thinkingBudgets?: ThinkingBudgets;
	/** Preferred transport forwarded to the stream function. */
	public transport: Transport;
	/** Optional cap for provider-requested retry delays. */
	public maxRetryDelayMs?: number;
	/** Tool execution strategy for assistant messages that contain multiple tool calls. */
	public toolExecution: ToolExecutionMode;

	constructor(options: AgentOptions) {
		// Older compiled consumers may omit options or streamFn even though the current API requires them.
		const runtimeOptions: Partial<AgentOptions> = options ?? {};
		this._state = createMutableAgentState(runtimeOptions.initialState);
		this.convertToLlm = runtimeOptions.convertToLlm ?? defaultConvertToLlm;
		this.transformContext = runtimeOptions.transformContext;
		this.streamFunction = runtimeOptions.streamFn ?? getDefaultStreamFn();
		this.getApiKey = runtimeOptions.getApiKey;
		this.onPayload = runtimeOptions.onPayload;
		this.onResponse = runtimeOptions.onResponse;
		this.onProviderStreamEvent = runtimeOptions.onProviderStreamEvent;
		this.beforeToolCall = runtimeOptions.beforeToolCall;
		this.afterToolCall = runtimeOptions.afterToolCall;
		this.finishTurn = runtimeOptions.finishTurn;
		this.prepareRequest = runtimeOptions.prepareRequest;
		this.prepareNextTurn = runtimeOptions.prepareNextTurn;
		this.prepareNextTurnWithContext = runtimeOptions.prepareNextTurnWithContext;
		this.steeringQueue = new PendingMessageQueue(runtimeOptions.steeringMode ?? "one-at-a-time");
		this.followUpQueue = new PendingMessageQueue(runtimeOptions.followUpMode ?? "one-at-a-time");
		this.sessionId = runtimeOptions.sessionId;
		this.thinkingBudgets = runtimeOptions.thinkingBudgets;
		this.transport = runtimeOptions.transport ?? "auto";
		this.maxRetryDelayMs = runtimeOptions.maxRetryDelayMs;
		this.toolExecution = runtimeOptions.toolExecution ?? "parallel";
	}

	/**
	 * Subscribe to agent lifecycle events.
	 *
	 * Listener promises are awaited in subscription order and are included in
	 * the current run's settlement. Listeners also receive the active abort
	 * signal for the current run.
	 *
	 * `agent_end` is the final emitted event for a run, but the agent does not
	 * become idle until all awaited listeners for that event have settled.
	 */
	subscribe(listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	/**
	 * Current agent state.
	 *
	 * Assigning `state.tools` or `state.messages` copies the provided top-level array.
	 */
	get state(): AgentState {
		return this._state;
	}

	/** Controls how queued steering messages are drained. */
	set steeringMode(mode: QueueMode) {
		this.steeringQueue.mode = mode;
	}

	get steeringMode(): QueueMode {
		return this.steeringQueue.mode;
	}

	/** Controls how queued follow-up messages are drained. */
	set followUpMode(mode: QueueMode) {
		this.followUpQueue.mode = mode;
	}

	get followUpMode(): QueueMode {
		return this.followUpQueue.mode;
	}

	/** Queue a message to be injected after the current assistant turn finishes. */
	steer(message: AgentMessage): void {
		this.steeringQueue.enqueue(message);
	}

	/** Queue a message to run only after the agent would otherwise stop. */
	followUp(message: AgentMessage): void {
		this.followUpQueue.enqueue(message);
	}

	/** Remove all queued steering messages. */
	clearSteeringQueue(): void {
		this.steeringQueue.clear();
	}

	/** Remove all queued follow-up messages. */
	clearFollowUpQueue(): void {
		this.followUpQueue.clear();
	}

	/** Remove all queued steering and follow-up messages. */
	clearAllQueues(): void {
		this.clearSteeringQueue();
		this.clearFollowUpQueue();
	}

	/** Returns true when either queue still contains pending messages. */
	hasQueuedMessages(): boolean {
		return this.steeringQueue.hasItems() || this.followUpQueue.hasItems();
	}

	/** Preview the messages selected for the next turn without consuming them. */
	peekQueuedMessages(): AgentMessage[] {
		const steering = this.steeringQueue.peek();
		return steering.length > 0 ? steering : this.followUpQueue.peek();
	}

	/** Active abort signal for the current run, if any. */
	get signal(): AbortSignal | undefined {
		return this.activeRun?.abortController.signal;
	}

	/** Abort the current run, if one is active. */
	abort(): void {
		this.activeRun?.abortController.abort();
	}

	/**
	 * Resolve when the current run and all awaited event listeners have finished.
	 *
	 * This resolves after `agent_end` listeners settle.
	 */
	waitForIdle(): Promise<void> {
		return this.activeRun?.promise ?? Promise.resolve();
	}

	/** Clear conversation state and queues while retaining the replayed prompt/tool baseline. */
	reset(): void {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before resetting.");
		}

		const baseline = getCurrentSystemMessage(this._state.messages);
		this._state.messages = baseline ? [baseline] : [];
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this._state.errorMessage = undefined;
		this.clearFollowUpQueue();
		this.clearSteeringQueue();
	}

	/** Start a new prompt from text, a single message, or a batch of messages. */
	async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
	async prompt(input: string, images?: ImageContent[]): Promise<void>;
	// 触发一次agent loop的入口方法
	async prompt(input: string | AgentMessage | AgentMessage[], images?: ImageContent[]): Promise<void> {
		// 如果当前agent已经在跑了，报错，只能使用steer或者followup来添加消息到队列中
		if (this.activeRun) {
			throw new Error(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
			);
		}
		// 规范化prompt的输入，将字符串、图片信息都转换成Message数组的形式返回
		const messages = this.normalizePromptInput(input, images);
		// 执行具体的prompt处理逻辑
		await this.runPromptMessages(messages);
	}

	/** Continue from the current transcript. The last message must be a user or tool-result message. */
	async continue(): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before continuing.");
		}

		const lastMessage = this._state.messages[this._state.messages.length - 1];
		if (!lastMessage || this._state.messages.every((message) => message.role === "system")) {
			throw new Error("No messages to continue from");
		}

		if (lastMessage.role === "assistant") {
			const queuedSteering = this.steeringQueue.drain();
			if (queuedSteering.length > 0) {
				await this.runPromptMessages(queuedSteering, { skipInitialSteeringPoll: true });
				return;
			}

			const queuedFollowUps = this.followUpQueue.drain();
			if (queuedFollowUps.length > 0) {
				await this.runPromptMessages(queuedFollowUps);
				return;
			}

			throw new Error("Cannot continue from message role: assistant");
		}

		await this.runContinuation();
	}

	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		// 如果input已经是数组了，直接返回
		if (Array.isArray(input)) {
			return input;
		}

		// 如果input不是直接输入的字符串，包装成数组返回
		if (typeof input !== "string") {
			return [input];
		}

		// 将输入的字符串包装成TextContent，然后将ImageContent也放入到content数组中
		const content: Array<TextContent | ImageContent> = [{ type: "text", text: input }];
		if (images && images.length > 0) {
			content.push(...images);
		}
		// 根据content构建一个UserMessage的数组返回
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	private async runPromptMessages(
		messages: AgentMessage[],
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoop(
				messages,
				// 创建agent loop要使用的上下文，里面就只有messages快照和tools快照
				this.createContextSnapshot(),
				// 根据options创建出agent loop要使用的config
				this.createLoopConfig(options),
				// 事件处理函数，在agent执行过程中可能会发送各种事件，该函数就是具体的处理逻辑
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	private async runContinuation(): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoopContinue(
				this.createContextSnapshot(),
				this.createLoopConfig(),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	private createContextSnapshot(): AgentContext {
		return {
			messages: this._state.messages.slice(),
			tools: this._state.tools.slice(),
		};
	}

	private createLoopConfig(options: { skipInitialSteeringPoll?: boolean } = {}): AgentLoopConfig {
		// 这里判断是否跳过初始化阶段的steer消息的拉取
		let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
		return {
			model: this._state.model,
			reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel,
			sessionId: this.sessionId,
			onPayload: this.onPayload,
			onResponse: this.onResponse,
			onProviderStreamEvent: this.onProviderStreamEvent,
			transport: this.transport,
			thinkingBudgets: this.thinkingBudgets,
			maxRetryDelayMs: this.maxRetryDelayMs,
			toolExecution: this.toolExecution,
			beforeToolCall: this.beforeToolCall,
			afterToolCall: this.afterToolCall,
			finishTurn: this.finishTurn,
			prepareRequest: this.prepareRequest,
			// 这个方法是在内部turn循环中准备下一次turn调用的
			prepareNextTurn:
				// agent里面的这两个方法是在构造agent的时候由AgentOptions传入的
				this.prepareNextTurnWithContext || this.prepareNextTurn
					? async (context) => {
							if (this.prepareNextTurnWithContext) {
								return await this.prepareNextTurnWithContext(context, this.signal);
							}
							return await this.prepareNextTurn?.(this.signal);
						}
					: undefined,
			convertToLlm: this.convertToLlm,
			transformContext: this.transformContext,
			getApiKey: this.getApiKey,
			// 获取steering message的方法
			getSteeringMessages: async () => {
				// 如果是跳过初始化steer消息的拉取，直接返回空数组，然后将skip标志设置为false，这样下次就不会跳过了
				if (skipInitialSteeringPoll) {
					skipInitialSteeringPoll = false;
					return [];
				}
				// 从队列中取出消息数组返回
				return this.steeringQueue.drain();
			},
			// 获取follow-up messages的方法
			getFollowUpMessages: async () => this.followUpQueue.drain(),
		};
	}

	private async runWithLifecycle(executor: (signal: AbortSignal) => Promise<void>): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing.");
		}

		// 创建一个abortController
		const abortController = new AbortController();
		let resolvePromise = () => {};
		// 创建一个Promise，并且将这个Promise的resolve函数给resolvePromise，用于将这个Promise设置为fulfilled状态
		// 这样做的目的是将resolve函数提取出来，以便别人来调用
		const promise = new Promise<void>((resolve) => {
			resolvePromise = resolve;
		});
		// 用上述三个元素创建一个activeRun对象，表示agent处于run状态，并且也留下了其他调用方完成或者中断promise的句柄
		this.activeRun = { promise, resolve: resolvePromise, abortController };

		// 修改agent持有的_state状态对象的属性
		this._state.isStreaming = true;
		this._state.streamingMessage = undefined;
		this._state.errorMessage = undefined;

		try {
			// 执行实际的executor，也就是agent loop的逻辑
			await executor(abortController.signal);
		} catch (error) {
			await this.handleRunFailure(error, abortController.signal.aborted);
		} finally {
			this.finishRun();
		}
	}

	private async handleRunFailure(error: unknown, aborted: boolean): Promise<void> {
		const failureMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? "aborted" : "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		} satisfies AgentMessage;
		await this.processEvents({ type: "message_start", message: failureMessage });
		await this.processEvents({ type: "message_end", message: failureMessage });
		await this.processEvents({ type: "turn_end", message: failureMessage, toolResults: [] });
		await this.processEvents({ type: "agent_end", messages: [failureMessage] });
	}

	private finishRun(): void {
		// 设置agentState状态 以及 调用activeRun的resolve方法，将对应的Promise设置为fulfilled
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this.activeRun?.resolve();
		this.activeRun = undefined;
	}

	/**
	 * Reduce internal state for a loop event, then await listeners.
	 *
	 * `agent_end` only means no further loop events will be emitted. The run is
	 * considered idle later, after all awaited listeners for `agent_end` finish
	 * and `finishRun()` clears runtime-owned state.
	 */
	private async processEvents(event: AgentEvent): Promise<void> {
		switch (event.type) {
			// 将事件里面带有的message放入到_state里面的streamingMessage字段
			case "message_start":
				this._state.streamingMessage = event.message;
				break;
			// 同上
			case "message_update":
				this._state.streamingMessage = event.message;
				break;
			// 将_state里面的streamingMessage设置为undefined，并且将事件里面的message放到state的message集合中
			case "message_end":
				this._state.streamingMessage = undefined;
				this._state.messages.push(event.message);
				break;
			// 维护state里面pendingToolCalls这个数组，将要待执行的toolCallId进行新增或者删除
			case "tool_execution_start": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.add(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "tool_execution_end": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.delete(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}
			// 一个 turn 指的是：一次请求模型，得到一条助手回复，再执行这条回复里的所有工具调用；一次prompt叫做run，run里面可能会有多个turn。
			// 如果事件里面带的message是ai输出的，并且存在errorMessage，将其维护进state的errorMessage中
			case "turn_end":
				if (event.message.role === "assistant" && event.message.errorMessage) {
					this._state.errorMessage = event.message.errorMessage;
				}
				break;
			// 表示一个agent loop结束了，将streamingMessage设置为undefined
			case "agent_end":
				this._state.streamingMessage = undefined;
				break;
		}

		// 获取当前activeRun里面的abortController的signal
		const signal = this.activeRun?.abortController.signal;
		// 如果不存在，报错
		if (!signal) {
			throw new Error("Agent listener invoked outside active run");
		}
		// 遍历所有的listener，推送事件和signal给listener
		for (const listener of this.listeners) {
			await listener(event, signal);
		}
	}
}
