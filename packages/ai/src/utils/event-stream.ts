import type { AssistantMessage, AssistantMessageEvent } from "../types.ts";

class FifoQueue<T> {
	// 使用两个数组的原因是将时间复杂度平均降低到O(1)
	// 如果使用一个数组的话，删除头元素之后会移动剩余的所有元素的位置，出队操作时间复杂度是O(n)
	private incoming: T[] = [];
	private outgoing: T[] = [];

	get length(): number {
		return this.incoming.length + this.outgoing.length;
	}

	// 入队放进incoming这个数组的末尾
	enqueue(value: T): void {
		this.incoming.push(value);
	}

	// 出队从outgoing数组末尾的位置选取元素删除并返回
	dequeue(): T | undefined {
		// 当outgoing没有元素的时候，从incoming末尾依次将元素删除并加入到outgoing末尾，这样就达成了先进先出的效果
		if (this.outgoing.length === 0) {
			while (this.incoming.length > 0) {
				this.outgoing.push(this.incoming.pop()!);
			}
		}
		// 从outgoing末尾删除并返回元素
		return this.outgoing.pop();
	}
}

// Generic event stream class for async iteration
export class EventStream<T, R = T> implements AsyncIterable<T> {
	// 创建一个先进先出的队列
	private queue = new FifoQueue<T>();
	// 创建一个等待的队列
	private waiting = new FifoQueue<(value: IteratorResult<T>) => void>();
	// 完成标志
	private done = false;
	private finalResultPromise: Promise<R>;
	private resolveFinalResult!: (result: R) => void;
	// 根据事件判断是否结束的函数
	private isComplete: (event: T) => boolean;
	// 根据事件提取结果的函数
	private extractResult: (event: T) => R;

	constructor(isComplete: (event: T) => boolean, extractResult: (event: T) => R) {
		this.isComplete = isComplete;
		this.extractResult = extractResult;
		// 创建一个promise，并将它的resolve方法赋值给resolveFinalResult，也就是说后续调用resolveFinalResult函数就会将这个promise置为fulfilled
		this.finalResultPromise = new Promise((resolve) => {
			this.resolveFinalResult = resolve;
		});
	}

	// 向stream中推送时间
	push(event: T): void {
		// 如果已经完成，直接返回
		if (this.done) return;

		// 判断是否是结束事件
		if (this.isComplete(event)) {
			// 如果是，将done设置为true
			this.done = true;
			// 从事件中提取结果，然后调用resolveFinalResult方法，将结果传入，并将finalResultPromise置为fulfilled
			this.resolveFinalResult(this.extractResult(event));
		}

		// Deliver to waiting consumer or queue it
		// 获取一个等待者
		const waiter = this.waiting.dequeue();
		// 如果存在等待着，调用等待函数
		if (waiter) {
			waiter({ value: event, done: false });
		} else {
			// 否则将事件入队
			this.queue.enqueue(event);
		}
	}

	// 手动将stream设置为完成状态
	end(result?: R): void {
		// 设置done为true
		this.done = true;
		if (result !== undefined) {
			this.resolveFinalResult(result);
		}
		// Notify all waiting consumers that we're done
		// 遍历waiting队列，向所有等待的promise发送结束事件
		while (this.waiting.length > 0) {
			const waiter = this.waiting.dequeue()!;
			waiter({ value: undefined as any, done: true });
		}
	}

	async *[Symbol.asyncIterator](): AsyncIterator<T> {
		while (true) {
			// 如果队列中有事件，出队，继续遍历
			if (this.queue.length > 0) {
				yield this.queue.dequeue()!;
			} else if (this.done) {
				// 如果整体stream已经完成了，直接返回
				return;
			} else {
				// 如果没有元素，创建一个promise进行等待，将该promise的resolve函数放入等待队列，当有事件入队时，该resolve会被调用，并且会获取到传入的IteratorResult
				const result = await new Promise<IteratorResult<T>>((resolve) => this.waiting.enqueue(resolve));
				// 如果是完成标志，直接返回
				if (result.done) return;
				// 否则返回事件，继续遍历
				yield result.value;
			}
		}
	}

	// 返回整个stream结束的promise
	result(): Promise<R> {
		return this.finalResultPromise;
	}
}

export class AssistantMessageEventStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") {
					return event.message;
				} else if (event.type === "error") {
					return event.error;
				}
				throw new Error("Unexpected event type for final result");
			},
		);
	}
}

/** Factory function for AssistantMessageEventStream (for use in extensions) */
export function createAssistantMessageEventStream(): AssistantMessageEventStream {
	return new AssistantMessageEventStream();
}
