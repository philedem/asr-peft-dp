import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const BACKEND_URL = process.env.VITE_RTSP_BACKEND_URL || 'http://rtsp-transcription-api:8000';

export const POST: RequestHandler = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/rtsp/clear`, {
      method: 'POST',
    });

    const data = await response.json();

    if (!response.ok) {
      return json(data, { status: response.status });
    }

    return json(data);
  } catch (error: any) {
    return json({ error: error.message }, { status: 500 });
  }
};
