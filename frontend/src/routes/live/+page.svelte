<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  // Use relative API routes that proxy to backend
  const API_BASE = '/api/rtsp';
  const WS_POLL_INTERVAL = 2000; // Poll every 2 seconds for new transcripts
  
  // State
  let rtspUrl = 'rtsp://rtsp-test-server:8554/test-stream';
  let segmentDuration = 10;
  let modelName = 'NbAiLab/nb-whisper-small';
  let useVad = true;  // Use Voice Activity Detection
  let silenceDuration = 2.0;  // Seconds of silence to trigger
  let isConnected = false;
  let isLoading = false;
  let transcripts: Array<{id: number, timestamp: string, text: string, duration: number}> = [];
  let error = '';
  let autoScroll = true;
  let showConfig = false;
  
  // Statistics
  let stats = {
    totalSegments: 0,
    avgTranscriptionTime: 0,
    startTime: null as Date | null
  };
  
  let pollInterval: number;
  let transcriptContainer: HTMLElement;
  
  function formatTimestamp(date: Date): string {
    return date.toLocaleTimeString('no-NO', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  }
  
  function getElapsedTime(): string {
    if (!stats.startTime) return '0:00';
    const diff = Date.now() - stats.startTime.getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }
  
  async function startTranscription() {
    isLoading = true;
    error = '';
    
    try {
      const response = await fetch(`${API_BASE}/start`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          rtsp_url: rtspUrl,
          segment_duration: segmentDuration,
          model_name: modelName,
          use_vad: useVad,
          silence_duration: silenceDuration
        })
      });
      
      if (!response.ok) {
        const data = await response.json();
        const errorMsg = data.detail || data.error || 'Failed to start transcription';
        
        // If already running, offer to stop and retry
        if (response.status === 400 && errorMsg.includes('already running')) {
          if (confirm('A transcription is already running. Stop it and start a new one?')) {
            await stopTranscription();
            // Retry starting
            setTimeout(() => startTranscription(), 500);
            return;
          }
        }
        
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      isConnected = true;
      stats.startTime = new Date();
      
      // Start polling for transcripts
      pollInterval = setInterval(fetchTranscripts, WS_POLL_INTERVAL);
      
    } catch (e: any) {
      error = e.message;
      isConnected = false;
    } finally {
      isLoading = false;
    }
  }
  
  async function stopTranscription() {
    isLoading = true;
    
    try {
      const response = await fetch(`${API_BASE}/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop transcription');
      }
      
      isConnected = false;
      if (pollInterval) clearInterval(pollInterval);
      
    } catch (e: any) {
      error = e.message;
    } finally {
      isLoading = false;
    }
  }
  
  async function fetchTranscripts() {
    try {
      const response = await fetch(`${API_BASE}/transcripts`);
      if (!response.ok) return;
      
      const data = await response.json();
      
      if (data.transcripts) {
        const hasNewTranscripts = data.transcripts.length > transcripts.length;
        // Reverse order - newest first
        transcripts = [...data.transcripts].reverse();
        stats.totalSegments = data.total || transcripts.length;
        stats.avgTranscriptionTime = data.avg_transcription_time || 0;
        
        // Auto-scroll to top for newest transcripts
        if (hasNewTranscripts && autoScroll && transcriptContainer) {
          setTimeout(() => {
            transcriptContainer.scrollTop = 0;
          }, 100);
        }
      }
      
    } catch (e) {
      console.error('Failed to fetch transcripts:', e);
    }
  }
  
  async function clearTranscripts() {
    if (confirm('Clear all transcripts?')) {
      try {
        const response = await fetch(`${API_BASE}/clear`, {
          method: 'POST'
        });
        
        if (!response.ok) {
          throw new Error('Failed to clear transcripts on backend');
        }
        
        transcripts = [];
        stats.totalSegments = 0;
        stats.avgTranscriptionTime = 0;
      } catch (e: any) {
        error = e.message;
      }
    }
  }
  
  function exportTranscripts() {
    const text = transcripts
      .map(t => `[${t.timestamp}] ${t.text}`)
      .join('\n\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcripts_${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  onMount(async () => {
    // Check backend status
    try {
      await fetch(`${API_BASE.replace('/api/rtsp', '')}/health`);
    } catch {
      error = 'Cannot connect to backend';
      return;
    }
    
    // Check if transcription is already running
    try {
      const response = await fetch(`${API_BASE}/transcripts`);
      if (response.ok) {
        const data = await response.json();
        if (data.is_running) {
          // Transcription is already running
          isConnected = true;
          transcripts = data.transcripts || [];
          stats.totalSegments = data.total || 0;
          stats.avgTranscriptionTime = data.avg_transcription_time || 0;
          stats.startTime = new Date(Date.now() - (data.elapsed_time || 0) * 1000);
          
          // Start polling
          pollInterval = setInterval(fetchTranscripts, WS_POLL_INTERVAL);
          
          console.log('Resumed existing transcription session');
        }
      }
    } catch (e) {
      console.error('Failed to check transcription status:', e);
    }
  });
  
  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
    if (isConnected) {
      fetch(`${BACKEND_URL}/rtsp/stop`, { method: 'POST' }).catch(console.error);
    }
  });
</script>

<svelte:head>
  <title>RTSP Live Transcription</title>
</svelte:head>

<div class="container">
  <header>
    <h1>🎙️ RTSP Live Transcription</h1>
    <p class="subtitle">Real-time speech transcription from RTSP streams</p>
  </header>

  <!-- Configuration Panel -->
  <div class="config-panel" class:collapsed={!showConfig}>
    <button class="toggle-config" on:click={() => showConfig = !showConfig}>
      {showConfig ? '▼' : '▶'} Configuration
    </button>
    
    {#if showConfig}
      <div class="config-content">
        <div class="form-group">
          <label for="rtsp-url">RTSP URL:</label>
          <input 
            id="rtsp-url"
            type="text" 
            bind:value={rtspUrl} 
            disabled={isConnected}
            placeholder="rtsp://server:port/stream"
          />
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label for="segment-duration">Segment Duration (seconds):</label>
            <input 
              id="segment-duration"
              type="number" 
              bind:value={segmentDuration} 
              disabled={isConnected}
              min="5"
              max="30"
            />
          </div>
          
          <div class="form-group">
            <label for="model-name">Whisper Model:</label>
            <select id="model-name" bind:value={modelName} disabled={isConnected}>
              <option value="NbAiLab/nb-whisper-tiny">nb-whisper-tiny (fast)</option>
              <option value="NbAiLab/nb-whisper-small">nb-whisper-small (balanced)</option>
              <option value="NbAiLab/nb-whisper-medium">nb-whisper-medium (accurate)</option>
            </select>
          </div>
        </div>
        
        <!-- VAD Settings -->
        <div class="vad-section">
          <label class="checkbox">
            <input type="checkbox" bind:checked={useVad} disabled={isConnected} />
            <span>Voice Activity Detection (VAD)</span>
          </label>
          <small class="help-text">
            {useVad ? `Transcribe on ${silenceDuration}s silence or ${segmentDuration}s max` : `Fixed ${segmentDuration}s segments`}
          </small>
        </div>
        
        {#if useVad}
          <div class="form-row">
            <div class="form-group">
              <label for="silence-duration">Silence Duration (seconds):</label>
              <input 
                id="silence-duration"
                type="number" 
                bind:value={silenceDuration} 
                disabled={isConnected}
                min="1"
                max="5"
                step="0.5"
              />
              <small class="help-text">Trigger transcription after this much silence</small>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Control Panel -->
  <div class="control-panel">
    <div class="controls">
      {#if !isConnected}
        <button 
          class="btn btn-primary" 
          on:click={startTranscription}
          disabled={isLoading || !rtspUrl}
        >
          {isLoading ? '⏳ Connecting...' : '▶️ Start Transcription'}
        </button>
      {:else}
        <button 
          class="btn btn-danger" 
          on:click={stopTranscription}
          disabled={isLoading}
        >
          {isLoading ? '⏳ Stopping...' : '⏹️ Stop'}
        </button>
      {/if}
      
      <button 
        class="btn btn-secondary" 
        on:click={clearTranscripts}
        disabled={transcripts.length === 0}
      >
        🗑️ Clear
      </button>
      
      <button 
        class="btn btn-secondary" 
        on:click={exportTranscripts}
        disabled={transcripts.length === 0}
      >
        💾 Export
      </button>
    </div>
    
    <div class="options">
      <label class="checkbox">
        <input type="checkbox" bind:checked={autoScroll} />
        Auto-scroll
      </label>
    </div>
  </div>

  <!-- Status Bar -->
  {#if isConnected}
    <div class="status-bar connected">
      <div class="status-item">
        <span class="status-label">🔴 Live</span>
      </div>
      <div class="status-item">
        <span class="status-label">Segments:</span>
        <span class="status-value">{stats.totalSegments}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Elapsed:</span>
        <span class="status-value">{getElapsedTime()}</span>
      </div>
      {#if stats.avgTranscriptionTime > 0}
        <div class="status-item">
          <span class="status-label">Avg Time:</span>
          <span class="status-value">{stats.avgTranscriptionTime.toFixed(1)}s</span>
        </div>
      {/if}
    </div>
  {:else if error}
    <div class="status-bar error">
      <span>❌ {error}</span>
    </div>
  {:else}
    <div class="status-bar idle">
      <span>⏸️ Idle - Configure and start transcription</span>
    </div>
  {/if}

  <!-- Transcripts Display -->
  <div class="transcripts-container" bind:this={transcriptContainer}>
    {#if transcripts.length === 0}
      <div class="empty-state">
        <p>No transcripts yet.</p>
        <p class="hint">Start transcription to see live results.</p>
      </div>
    {:else}
      {#each transcripts as transcript (transcript.id)}
        <div class="transcript-item">
          <div class="transcript-header">
            <span class="transcript-id">#{transcript.id}</span>
            <span class="transcript-time">{transcript.timestamp}</span>
            {#if transcript.duration}
              <span class="transcript-duration">⏱️ {transcript.duration.toFixed(1)}s</span>
            {/if}
          </div>
          <div class="transcript-text">{transcript.text}</div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');

  :global(body) {
    font-family: 'Roboto Mono', 'Courier New', monospace;
    margin: 0;
    padding: 20px;
    background: #0f1419;
    min-height: 100vh;
    color: #e8ecef;
  }

  .container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0;
  }

  header {
    text-align: center;
    margin-bottom: 2rem;
  }

  h1 {
    color: #00d4aa;
    font-weight: 700;
    margin-bottom: 0.5em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 2em;
    border-bottom: 2px solid #00d4aa;
    padding-bottom: 0.5em;
  }

  .subtitle {
    text-align: center;
    color: #a8b2bb;
    margin-top: -0.5em;
    margin-bottom: 2em;
    font-size: 0.95em;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Configuration Panel */
  .config-panel {
    background: #1a1f26;
    border: 2px solid #2d3339;
    padding: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.1);
  }

  .config-panel.collapsed {
    padding: 0.5rem 1rem;
  }

  .toggle-config {
    background: none;
    border: none;
    color: #00d4aa;
    cursor: pointer;
    font-family: inherit;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.5rem 0;
    width: 100%;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .toggle-config:hover {
    color: #00ffcc;
  }

  .config-content {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #2d3339;
  }

  .form-group {
    margin-bottom: 1rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    color: #00d4aa;
    font-size: 0.85em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  input[type="text"],
  input[type="number"],
  select {
    width: 100%;
    padding: 0.75rem;
    background: #0f1419;
    border: 2px solid #2d3339;
    border-radius: 0;
    color: #e8ecef;
    font-family: inherit;
    font-size: 0.95rem;
    transition: all 0.2s;
  }

  input:focus,
  select:focus {
    outline: none;
    border-color: #00d4aa;
    box-shadow: 0 0 10px rgba(0, 212, 170, 0.2);
  }

  input:disabled,
  select:disabled {
    background: #2d3339;
    cursor: not-allowed;
    opacity: 0.5;
  }

  .vad-section {
    margin-top: 1.5rem;
    padding: 1rem;
    background: rgba(0, 212, 170, 0.05);
    border-left: 3px solid #00d4aa;
  }

  .vad-section .checkbox {
    font-size: 1em;
    margin-bottom: 0.5rem;
  }

  .vad-section .checkbox span {
    color: #e8ecef;
    font-weight: 600;
  }

  .help-text {
    display: block;
    color: #4a5159;
    font-size: 0.75em;
    margin-top: 0.25rem;
    text-transform: none;
    letter-spacing: 0;
  }

  /* Control Panel */
  .control-panel {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    background: #1a1f26;
    border: 2px solid #2d3339;
    padding: 1.5em;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.1);
  }

  .controls {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .btn {
    background: #00d4aa;
    color: #0f1419;
    border: 2px solid #00d4aa;
    border-radius: 0;
    font-size: 0.9em;
    font-weight: 700;
    padding: 0.8em 1.5em;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Roboto Mono', monospace;
  }

  .btn:hover:not(:disabled) {
    background: transparent;
    color: #00d4aa;
    box-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
  }

  .btn:disabled {
    background: #2d3339;
    border-color: #2d3339;
    color: #a8b2bb;
    cursor: not-allowed;
    opacity: 0.5;
  }

  .btn-primary {
    background: #00d4aa;
    border-color: #00d4aa;
    color: #0f1419;
  }

  .btn-primary:hover:not(:disabled) {
    background: transparent;
    color: #00d4aa;
  }

  .btn-danger {
    background: #ef5350;
    border-color: #ef5350;
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: transparent;
    color: #ef5350;
    box-shadow: 0 0 10px rgba(239, 83, 80, 0.3);
  }

  .btn-secondary {
    background: transparent;
    border: 2px solid #4a5159;
    color: #a8b2bb;
  }

  .btn-secondary:hover:not(:disabled) {
    border-color: #00d4aa;
    color: #00d4aa;
  }

  .options {
    display: flex;
    gap: 1rem;
  }

  .checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    color: #a8b2bb;
    text-transform: uppercase;
    font-size: 0.85em;
    letter-spacing: 0.05em;
  }

  .checkbox input {
    cursor: pointer;
  }

  /* Status Bar */
  .status-bar {
    padding: 1em 1.5em;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    flex-wrap: wrap;
    border: 2px solid #2d3339;
    font-size: 0.9em;
  }

  .status-bar.connected {
    background: #1a1f26;
    border-left: 4px solid #00d4aa;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.2);
    animation: pulse-border 2s ease-in-out infinite;
  }

  @keyframes pulse-border {
    0%, 100% { border-left-color: #00d4aa; }
    50% { border-left-color: rgba(0, 212, 170, 0.5); }
  }

  .status-bar.error {
    background: rgba(239, 83, 80, 0.1);
    border-left: 4px solid #ef5350;
    color: #ef5350;
  }

  .status-bar.idle {
    background: #1a1f26;
    border-left: 4px solid #4a5159;
    color: #a8b2bb;
  }

  .status-item {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .status-label {
    color: #00d4aa;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.85em;
    letter-spacing: 0.05em;
  }

  .status-value {
    color: #e8ecef;
    font-weight: 700;
  }

  /* Transcripts */
  .transcripts-container {
    background: #1a1f26;
    border: 2px solid #2d3339;
    padding: 1.5rem;
    max-height: 600px;
    overflow-y: auto;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.1);
  }

  .transcripts-container::-webkit-scrollbar {
    width: 8px;
  }

  .transcripts-container::-webkit-scrollbar-track {
    background: #0f1419;
  }

  .transcripts-container::-webkit-scrollbar-thumb {
    background: #2d3339;
  }

  .transcripts-container::-webkit-scrollbar-thumb:hover {
    background: #00d4aa;
  }

  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #a8b2bb;
  }

  .empty-state p {
    margin: 0.5rem 0;
  }

  .empty-state .hint {
    font-size: 0.85em;
    color: #4a5159;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .transcript-item {
    background: #0f1419;
    border: 1px solid #2d3339;
    border-left: 3px solid #00d4aa;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
  }

  .transcript-item:hover {
    background: rgba(0, 212, 170, 0.05);
  }

  .transcript-item:last-child {
    margin-bottom: 0;
  }

  .transcript-header {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 0.75rem;
    font-size: 0.75em;
    color: #a8b2bb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .transcript-id {
    color: #00d4aa;
    font-weight: 700;
  }

  .transcript-time {
    color: #4a5159;
    font-family: 'Roboto Mono', monospace;
  }

  .transcript-duration {
    color: #4a5159;
    margin-left: auto;
  }

  .transcript-text {
    color: #e8ecef;
    line-height: 1.6;
    font-size: 1rem;
  }

  @media (max-width: 768px) {
    :global(body) {
      padding: 10px;
    }

    h1 {
      font-size: 1.5em;
    }

    .form-row {
      grid-template-columns: 1fr;
    }

    .control-panel {
      flex-direction: column;
      align-items: stretch;
    }

    .controls {
      flex-direction: column;
    }

    .btn {
      width: 100%;
    }

    .status-bar {
      gap: 1rem;
    }

    .status-item {
      flex-direction: column;
      gap: 0.25rem;
      align-items: flex-start;
    }
  }
</style>
