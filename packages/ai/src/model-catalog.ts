import type {
	Api,
	ClassifierApi,
	ClassifierModel,
	ImageApi,
	ImageModel,
	Model,
	ModelType,
	ProviderId,
} from "./types.ts";

export type ModelGroups = Record<string, Record<string, object>>;

type ModelKey<TGroups extends ModelGroups> = {
	[TApi in keyof TGroups]: keyof TGroups[TApi];
}[keyof TGroups] &
	string;

type KeyForType<TGroups extends ModelGroups, TType extends ModelType> = Extract<
	ModelKey<TGroups>,
	`${TType}:${string}`
>;

type ModelId<TKey extends string> = TKey extends `${ModelType}:${infer TModelId}` ? TModelId : never;

type ApiForKey<TGroups extends ModelGroups, TKey extends ModelKey<TGroups>> = {
	[TApi in keyof TGroups]: TKey extends keyof TGroups[TApi] ? TApi : never;
}[keyof TGroups];

export type ChatModelCatalog<TGroups extends ModelGroups, TProvider extends ProviderId> = {
	[TKey in KeyForType<TGroups, "chat"> as ModelId<TKey>]: Model<ApiForKey<TGroups, TKey> & Api> & {
		id: ModelId<TKey>;
		provider: TProvider;
	};
};

export type ImageModelCatalog<TGroups extends ModelGroups, TProvider extends ProviderId> = {
	[TKey in KeyForType<TGroups, "image"> as ModelId<TKey>]: ImageModel<ApiForKey<TGroups, TKey> & ImageApi> & {
		id: ModelId<TKey>;
		provider: TProvider;
	};
};

export type ClassifierModelCatalog<TGroups extends ModelGroups, TProvider extends ProviderId> = {
	[TKey in KeyForType<TGroups, "classifier"> as ModelId<TKey>]: ClassifierModel<
		ApiForKey<TGroups, TKey> & ClassifierApi
	> & {
		id: ModelId<TKey>;
		provider: TProvider;
	};
};

function flattenModelCatalog(groups: ModelGroups, type: ModelType): Record<string, object> {
	return Object.fromEntries(
		// 这一步是按照api进行分组
		Object.values(groups)
			// 将每个api下的各种模型平铺成一个模型数组
			.flatMap((models) => Object.values(models))
			// 只筛选出对应类型的模型，比如chat类型的
			.filter((model) => (model as { type?: unknown }).type === type)
			// 将模型数组转换成 [id, model]格式的元组数组。最终通过Object.fromEntries组合成属性为各个id:model的对象
			.map((model) => [(model as { id: string }).id, model]),
	);
}

export function flattenChatModelCatalog<const TProvider extends ProviderId, const TGroups extends ModelGroups>(
	_provider: TProvider,
	groups: TGroups,
): ChatModelCatalog<TGroups, TProvider> {
	return flattenModelCatalog(groups, "chat") as ChatModelCatalog<TGroups, TProvider>;
}

export function flattenImageModelCatalog<const TProvider extends ProviderId, const TGroups extends ModelGroups>(
	_provider: TProvider,
	groups: TGroups,
): ImageModelCatalog<TGroups, TProvider> {
	return flattenModelCatalog(groups, "image") as ImageModelCatalog<TGroups, TProvider>;
}

export function flattenClassifierModelCatalog<const TProvider extends ProviderId, const TGroups extends ModelGroups>(
	_provider: TProvider,
	groups: TGroups,
): ClassifierModelCatalog<TGroups, TProvider> {
	return flattenModelCatalog(groups, "classifier") as ClassifierModelCatalog<TGroups, TProvider>;
}
