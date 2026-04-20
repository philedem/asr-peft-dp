import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0',
		allowedHosts: [
			'asr.192.168.3.155.nip.io',
			'asr.midlaier.local',
			'.cisk.unclassified.mil.no',
			'.local',
			'.nip.io',
			'localhost'
		]
	}
});
