// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_BACKEND_URL?: string;
	readonly VITE_WER_POLL_INTERVAL?: string;
	readonly VITE_WHISPERLIVE_WS_URL?: string;
	readonly VITE_ENABLE_TRAINING?: string;
	readonly VITE_ENABLE_STREAMING?: string;
	readonly VITE_ENABLE_BATCH_UPLOAD?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}

declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}
}

export {};
