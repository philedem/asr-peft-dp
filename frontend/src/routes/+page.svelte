<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fly }     from 'svelte/transition';

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const WER_POLL_INTERVAL = import.meta.env.VITE_WER_POLL_INTERVAL || 70000;
  const WHISPERLIVE_WS_URL = import.meta.env.VITE_WHISPERLIVE_WS_URL || 'ws://localhost:9090';
  
  // Feature flags
  const ENABLE_TRAINING = import.meta.env.VITE_ENABLE_TRAINING !== 'false';
  const ENABLE_STREAMING = import.meta.env.VITE_ENABLE_STREAMING === 'true';
  const ENABLE_BATCH_UPLOAD = import.meta.env.VITE_ENABLE_BATCH_UPLOAD !== 'false';

  let records:any[] = [];
  let loading=false, err='';
  let notification='', notifType='success', showNotif=false;
  let manual:Record<string,string>={}, reviewed:Record<string,boolean>={}, saving:Record<string,boolean>={};
  let originalManual:Record<string,string>={}; // Track original values to detect changes
  let wer='N/A';
  let modelInfo:any = null;
  let trainingStatus:any = {status: 'idle', progress: 0, message: ''};
  let isRecording=false, chunks:Blob[]=[], recorder:MediaRecorder|null=null;
  let uploadProgress=false;
  let trainingInProgress=false;
  let audioDevices:MediaDeviceInfo[]=[];
  let selectedDeviceId:string='';

  // WhisperLive streaming state
  let isStreaming=false, streamingWS:WebSocket|null=null;
  let streamRecorder:MediaRecorder|null=null, streamAudioContext:AudioContext|null=null;
  let currentStreamTranscript='', streamingChunks:Blob[]=[];
  let streamingStats:any = null;
  let rtspUrl:string = '';
  let useRTSP:boolean = false;
  let lastWSMessage:string = ''; // Debug: last WebSocket message received

  const toast=(m:string,t='success')=>{
    notification=m; notifType=t; showNotif=true;
    setTimeout(()=>showNotif=false,3000);
  };

  // ------- API helpers ------------
  const getJSON = async (u:string)=>{
    try {
      const response = await fetch(u);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch(e) {
      toast(`API Error: ${e}`, 'error');
      throw e;
    }
  };
  const post = async (u:string,body:any)=>{
    try {
      const response = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch(e) {
      toast(`API Error: ${e}`, 'error');
      throw e;
    }
  };

  async function fetchWER(){ 
    try {
      const endpoint = ENABLE_STREAMING ? `${BACKEND_URL}/corrections/stats` : `${BACKEND_URL}/asr/wer`;
      const data = await getJSON(endpoint);
      if (ENABLE_STREAMING) {
        wer = data.average_wer != null ? data.average_wer.toFixed(2) + '%' : 'N/A';
        streamingStats = data;
      } else {
        wer = data.wer || 'N/A';
      }
    } catch(e) {
      console.error('Failed to fetch WER:', e);
    }
  }
  
  async function fetchModelInfo(){
    try {
      modelInfo = await getJSON(`${BACKEND_URL}/asr/model_info`);
    } catch(e) {
      console.error('Failed to fetch model info:', e);
    }
  }
  
  async function fetchTrainingStatus(){
    try {
      const status = await getJSON(`${BACKEND_URL}/train/status`);
      const wasTraining = trainingInProgress;
      trainingInProgress = status.status === 'running';
      trainingStatus = status;
      
      // Show notification when training completes
      if (wasTraining && status.status === 'completed') {
        toast('✅ Training completed successfully!', 'success');
        await fetchWER();
        await fetchModelInfo();
        await load();
      } else if (wasTraining && status.status === 'failed') {
        toast('❌ Training failed', 'error');
      }
    } catch(e) {
      console.error('Failed to fetch training status:', e);
    }
  }
  
  async function load(){
    loading=true; err='';
    try{
      records=await getJSON(`${BACKEND_URL}/asr/records`);
      records.sort((a,b)=>(b.timestamp??'').localeCompare(a.timestamp??''));
      manual={}; reviewed={}; originalManual={};
      records.forEach(r=>{
        manual[r.audio_file] = r.manual_transcript?.length ? r.manual_transcript : r.asr_transcript || '';
        originalManual[r.audio_file] = manual[r.audio_file]; // Store original
        reviewed[r.audio_file] = !!(r.manual_transcript && r.manual_transcript.length > 0);
      });
    }catch(e){
      err='Could not load records. Is the backend running?';
      console.error(e);
    }
    loading=false;
  }

  async function save(r, force=false){
    if (saving[r.audio_file]) return;
    
    // Don't save if nothing changed (unless forced, e.g. from approve)
    if (!force && manual[r.audio_file] === originalManual[r.audio_file]) {
      return;
    }
    
    saving[r.audio_file]=true;
    try {
      await post(`${BACKEND_URL}/asr/save_record`,{
        audio_id:r.audio_file,
        asr_transcript:r.asr_transcript,
        manual_transcript:manual[r.audio_file],
      });
      reviewed[r.audio_file] = !!(manual[r.audio_file] && manual[r.audio_file].length > 0);
      originalManual[r.audio_file] = manual[r.audio_file]; // Update original after save
      toast('Saved successfully'); 
      await load(); 
      await fetchWER();
    } catch(e) {
      toast('Save failed', 'error');
    } finally {
      saving[r.audio_file]=false;
    }
  }
  
  // Check if there are unsaved changes
  const hasUnsavedChanges = (audioFile:string) => {
    return manual[audioFile] !== originalManual[audioFile];
  };

  // Delete a record
  async function deleteRecord(r:any) {
    if (!confirm(`Delete this record? This cannot be undone.\n\nAudio: ${r.audio_file}`)) {
      return;
    }
    
    try {
      const recordId = r.audio_file.replace('.wav', '');
      const res = await fetch(`${BACKEND_URL}/asr/records/${recordId}`, {
        method: 'DELETE'
      });
      
      if (!res.ok) {
        throw new Error('Delete failed');
      }
      
      toast('Record deleted');
      await load(); // Reload the records list
    } catch(e) {
      toast('Delete failed', 'error');
    }
  }

  // ---------- audio device enumeration ----------
  async function loadAudioDevices(){
    try {
      // Request permission first so labels are populated
      const tempStream=await navigator.mediaDevices.getUserMedia({audio:true});
      tempStream.getTracks().forEach(t=>t.stop());
      const devices=await navigator.mediaDevices.enumerateDevices();
      audioDevices=devices.filter(d=>d.kind==='audioinput');
      // Keep current selection if still valid, otherwise pick default
      if(!selectedDeviceId || !audioDevices.find(d=>d.deviceId===selectedDeviceId)){
        selectedDeviceId=audioDevices[0]?.deviceId||'';
      }
    } catch(e){
      console.error('Could not enumerate audio devices:',e);
    }
  }

  // ---------- recording ----------
  async function toggleRec(){
    if(!isRecording){
      try {
        chunks=[]; isRecording=true;
        const audioConstraints = selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId } }
          : {};
        const stream=await navigator.mediaDevices.getUserMedia({audio:audioConstraints});
        recorder=new MediaRecorder(stream);
        recorder.ondataavailable=e=>chunks.push(e.data);
        recorder.onstop=async ()=>{
          isRecording=false;
          await sendToASR(new Blob(chunks,{type:'audio/webm'}));
        };
        recorder.start();
        toast('Recording started...');
      } catch(e) {
        toast('Microphone access denied', 'error');
        isRecording=false;
      }
    }else {
      recorder?.stop();
      toast('Recording stopped');
    }
  }
  
  async function sendToASR(blob:Blob){
    uploadProgress=true;
    try {
      const fd=new FormData(); 
      fd.append('audio',blob,'record.webm');
      toast('Processing audio...');
      const r=await fetch(`${BACKEND_URL}/asr/transcribe`,{method:'POST',body:fd});
      if (r.ok) {
        const result = await r.json();
        toast(`✓ Created ${result.created} transcription(s)`);
      } else {
        const errorData = await r.json().catch(() => ({ error: 'Unknown error' }));
        toast(`Transcription failed: ${errorData.error}`, 'error');
      }
      await load(); 
      await fetchWER();
    } catch(e) {
      toast('Upload/transcription failed', 'error');
      console.error(e);
    } finally {
      uploadProgress=false;
    }
  }
  
  const approve=async (r)=>{ 
    manual[r.audio_file]=r.asr_transcript; 
    await save(r, true); // Force save even if value hasn't changed
  };

  // ---------- WhisperLive Streaming ----------
  async function toggleStreaming() {
    console.log('toggleStreaming called, isStreaming:', isStreaming);
    if (!isStreaming) {
      await startStreaming();
    } else {
      stopStreaming();
    }
  }

  async function startStreaming() {
    console.log('startStreaming called, useRTSP:', useRTSP, 'rtspUrl:', rtspUrl);
    try {
      currentStreamTranscript = '';
      streamingChunks = [];
      
      // If RTSP is selected, send URL to backend for processing
      if (useRTSP && rtspUrl.trim()) {
        await startRTSPStreaming();
        return;
      }
      
      // Otherwise use microphone (original behavior)
      await startMicrophoneStreaming();
    } catch (e) {
      console.error('Failed to start streaming:', e);
      toast('Failed to start streaming: ' + e.message, 'error');
      stopStreaming();
    }
  }

  async function startRTSPStreaming() {
    // Connect to WhisperLive WebSocket
    streamingWS = new WebSocket(WHISPERLIVE_WS_URL);
    
    streamingWS.onopen = () => {
      console.log('Connected to WhisperLive server for RTSP');
      
      // Send RTSP stream config
      const config = {
        uid: 'rtsp-client-' + Date.now(),
        language: 'no',
        task: 'transcribe',
        model: 'NbAiLab/nb-whisper-medium',
        use_vad: false,  // Disabled VAD temporarily - it was filtering out all audio
        device: 'cuda',
        compute_type: 'float16',
        rtsp_url: rtspUrl
      };
      streamingWS?.send(JSON.stringify(config));
      isStreaming = true;
      toast('🎥 RTSP streaming started', 'success');
    };
    
    streamingWS.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📩 [RTSP] Received WebSocket message:', data);
        
        if (data.message === 'SERVER_READY') {
          console.log('✅ Server is ready');
        } else if (data.segments && data.segments.length > 0) {
          console.log('📝 [RTSP] Transcription segments:', data.segments);
          currentStreamTranscript = data.segments.map(s => s.text).join(' ');
          console.log('[RTSP] Updated transcript:', currentStreamTranscript);
        } else {
          console.log('ℹ️ [RTSP] Message without segments:', data);
        }
      } catch (e) {
        console.error('[RTSP] Error parsing WebSocket message:', e, 'Raw data:', event.data);
      }
    };
    
    streamingWS.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast('RTSP streaming connection error', 'error');
      stopStreaming();
    };
    
    streamingWS.onclose = () => {
      console.log('WebSocket closed');
      if (isStreaming) {
        stopStreaming();
      }
    };
  }

  async function startMicrophoneStreaming() {
    // Connect to WhisperLive WebSocket
    streamingWS = new WebSocket(WHISPERLIVE_WS_URL);
    
    streamingWS.onopen = async () => {
      console.log('Connected to WhisperLive server');
      
      // Send connection config
      const config = {
        uid: 'web-client-' + Date.now(),
        language: 'no', // Norwegian
        task: 'transcribe',
        model: 'NbAiLab/nb-whisper-medium',
        use_vad: false,  // Disabled VAD temporarily - it was filtering out all audio
        device: 'cuda',
        compute_type: 'float16'
      };
      streamingWS?.send(JSON.stringify(config));
      
      // Start capturing audio with proper format for WhisperLive
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      streamAudioContext = new AudioContext({ sampleRate: 16000 });
      const source = streamAudioContext.createMediaStreamSource(stream);
      
      // Create ScriptProcessor to get raw PCM audio data
      const processor = streamAudioContext.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (e) => {
        if (streamingWS?.readyState === WebSocket.OPEN) {
          // Get raw PCM float32 audio data
          const inputData = e.inputBuffer.getChannelData(0);
          
          // Convert float32 to int16 PCM (what WhisperLive expects)
          const int16Data = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          
          // Send raw PCM data
          streamingWS?.send(int16Data.buffer);
        }
      };
      
      source.connect(processor);
      processor.connect(streamAudioContext.destination);
      
      // Store processor for cleanup
      (window as any).audioProcessor = processor;
      
      isStreaming = true;
      toast('🎙️ Streaming started', 'success');
    };
    
    streamingWS.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📩 Received WebSocket message:', data);
        lastWSMessage = JSON.stringify(data, null, 2); // Store for debug display
        
        if (data.message === 'SERVER_READY') {
          console.log('✅ Server is ready');
        } else if (data.segments && data.segments.length > 0) {
          console.log('📝 Transcription segments:', data.segments);
          // Update transcript with latest segments
          currentStreamTranscript = data.segments.map(s => s.text).join(' ');
          console.log('Updated transcript:', currentStreamTranscript);
        } else {
          console.log('ℹ️ Message without segments:', data);
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e, 'Raw data:', event.data);
        lastWSMessage = 'Parse error: ' + event.data;
      }
    };
    
    streamingWS.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast('Streaming connection error', 'error');
      stopStreaming();
    };
    
    streamingWS.onclose = () => {
      console.log('WebSocket closed');
      if (isStreaming) {
        stopStreaming();
      }
    };
  }

  function stopStreaming() {
    isStreaming = false;
    
    // Cleanup audio processor
    if ((window as any).audioProcessor) {
      (window as any).audioProcessor.disconnect();
      (window as any).audioProcessor = null;
    }
    
    if (streamRecorder) {
      streamRecorder.stop();
      streamRecorder.stream.getTracks().forEach(track => track.stop());
      streamRecorder = null;
    }
    
    if (streamAudioContext) {
      streamAudioContext.close();
      streamAudioContext = null;
    }
    
    if (streamingWS) {
      streamingWS.close();
      streamingWS = null;
    }
    
    toast('🛑 Streaming stopped', 'success');
  }

  async function saveStreamCorrection() {
    if (!currentStreamTranscript.trim()) {
      toast('No transcript to save', 'error');
      return;
    }
    
    const correctedText = prompt('Edit the transcription if needed:', currentStreamTranscript);
    if (correctedText === null) return; // User cancelled
    
    try {
      await post(`${BACKEND_URL}/corrections/save`, {
        original_transcript: currentStreamTranscript,
        corrected_transcript: correctedText
      });
      
      toast('✅ Correction saved', 'success');
      currentStreamTranscript = '';
      await fetchWER();
    } catch (e) {
      toast('Failed to save correction', 'error');
    }
  }

  async function triggerManualRetrain() {
    if (trainingInProgress) return;
    trainingInProgress = true;
    toast('Starting training... This may take several minutes.', 'success');
    try {
      await getJSON(`${BACKEND_URL}/train/retrain_lora`);
      // Don't set trainingInProgress to false - let the status polling handle it
    } catch(e) {
      toast('Failed to start training', 'error');
      trainingInProgress = false;
    }
  }

  async function calculateWEROnly() {
    if (trainingInProgress) return;
    
    // Check if there are any reviewed records
    const reviewedCount = Object.values(reviewed).filter(Boolean).length;
    if (reviewedCount === 0) {
      toast('No reviewed records to calculate WER. Please review at least one record.', 'error');
      return;
    }
    
    trainingInProgress = true;
    toast('Calculating WER...', 'success');
    try {
      await getJSON(`${BACKEND_URL}/train/calculate_wer`);
      
      // Poll for completion and update WER
      const pollCompletion = async () => {
        for (let i = 0; i < 20; i++) { // Poll for up to 10 seconds
          await new Promise(resolve => setTimeout(resolve, 500));
          const status = await getJSON(`${BACKEND_URL}/train/status`);
          if (status.status === 'completed') {
            trainingInProgress = false;
            await fetchWER();
            toast('✅ WER calculation completed!', 'success');
            return;
          } else if (status.status === 'failed') {
            trainingInProgress = false;
            toast('❌ WER calculation failed', 'error');
            return;
          }
        }
        // Fallback: assume it completed if we timed out
        trainingInProgress = false;
        await fetchWER();
      };
      pollCompletion();
      
    } catch(e) {
      toast('Failed to start WER calculation', 'error');
      trainingInProgress = false;
    }
  }

  onMount(()=>{ 
    load(); 
    loadAudioDevices();
    navigator.mediaDevices?.addEventListener('devicechange', loadAudioDevices);
    if (!ENABLE_STREAMING) {
      load(); 
    }
    fetchWER();
    if (ENABLE_TRAINING) {
      fetchModelInfo();
      fetchTrainingStatus();
      setInterval(fetchModelInfo, WER_POLL_INTERVAL);
      setInterval(fetchTrainingStatus, 5000);
    }
    setInterval(fetchWER, WER_POLL_INTERVAL);
  });

  onDestroy(() => {
    if (isStreaming) {
      stopStreaming();
    }
  });

  const fmt=t=>t?.replace('T',' ').replace('Z','').slice(0,19);
</script>

<h1>ASR Annotation System</h1>
<p class="subtitle">{ENABLE_STREAMING ? 'Real-time Streaming & Corrections' : 'Transcription & Model Training'}</p>

{#if ENABLE_TRAINING && modelInfo}
  <div class="model-info-banner">
    <span class="model-name">🤖 Model: <strong>{modelInfo.base_model}</strong></span>
    <span class="model-status">
      {#if modelInfo.has_lora_adapter}
        ✅ LoRA Fine-tuned {modelInfo.training_iteration ? `(Iteration #${modelInfo.training_iteration})` : ''}
      {:else}
        🔵 Base Model (No Fine-tuning)
      {/if}
    </span>
    <span class="device-info">💻 {modelInfo.device.toUpperCase()}</span>
  </div>
{/if}

<div class="control-panel">
  <div class="audio-section">
    <h3>Audio Input</h3>
    {#if audioDevices.length > 0}
      <select class="device-select" bind:value={selectedDeviceId} disabled={isRecording || uploadProgress}>
        {#each audioDevices as dev}
          <option value={dev.deviceId}>{dev.label || `Microphone ${audioDevices.indexOf(dev)+1}`}</option>
        {/each}
      </select>
    {/if}
    <button class="button record-btn" class:recording={isRecording} on:click={toggleRec} disabled={uploadProgress}>
      {isRecording ? '⏹ Stop Recording' : '🎤 Start Recording'}
    </button>
    <label class="button upload-btn">
      📁 Upload Audio
      <input type="file" accept="audio/*" style="display:none" on:change={e=>sendToASR(e.target.files[0])} disabled={uploadProgress}/>
    </label>
    {#if uploadProgress}
      <span class="status-indicator">⏳ Processing...</span>
    {/if}
  </div>
  {#if ENABLE_STREAMING}
    <!-- WhisperLive Streaming Mode -->
    <div class="streaming-section">
      <h3>🎙️ Live Streaming</h3>
      
      <!-- Input method selector -->
      <div class="input-selector">
        <label>
          <input type="radio" bind:group={useRTSP} value={false} disabled={isStreaming} />
          🎤 Microphone
        </label>
        <label>
          <input type="radio" bind:group={useRTSP} value={true} disabled={isStreaming} />
          🎥 RTSP Stream
        </label>
      </div>
      
      <!-- RTSP URL input -->
      {#if useRTSP}
        <div class="rtsp-input">
          <input 
            type="text" 
            bind:value={rtspUrl} 
            placeholder="rtsp://username:password@camera-ip:554/stream"
            disabled={isStreaming}
            class="rtsp-url-input"
          />
          <small class="hint">Example: rtsp://admin:password@192.168.1.100:554/stream1</small>
        </div>
      {/if}
      
      <button 
        class="button stream-btn" 
        class:streaming={isStreaming} 
        on:click={toggleStreaming}
        disabled={useRTSP && !rtspUrl.trim() && !isStreaming}
      >
        {isStreaming ? '⏹ Stop Streaming' : '▶️ Start Streaming'}
      </button>
      
      {#if isStreaming}
        <div class="streaming-status">
          <strong>🎤 Streaming Active...</strong>
          <p class="status-hint">Speak clearly into your microphone. Transcription will appear below.</p>
        </div>
      {/if}
      
      {#if lastWSMessage}
        <details class="debug-section">
          <summary>🔍 Debug: Last WebSocket Message</summary>
          <pre>{lastWSMessage}</pre>
        </details>
      {/if}
      
      {#if currentStreamTranscript}
        <div class="stream-transcript">
          <strong>Current Transcript:</strong>
          <p>{currentStreamTranscript}</p>
          <button class="button save-btn" on:click={saveStreamCorrection}>
            💾 Save & Correct
          </button>
        </div>
      {/if}
    </div>
  {:else if ENABLE_BATCH_UPLOAD}
    <!-- Batch Upload Mode -->
    <div class="audio-section">
      <h3>Audio Input</h3>
      <button class="button record-btn" class:recording={isRecording} on:click={toggleRec} disabled={uploadProgress}>
        {isRecording ? '⏹ Stop Recording' : '🎤 Start Recording'}
      </button>
      <label class="button upload-btn">
        📁 Upload Audio
        <input type="file" accept="audio/*" style="display:none" on:change={e=>sendToASR(e.target.files[0])} disabled={uploadProgress}/>
      </label>
      {#if uploadProgress}
        <span class="status-indicator">⏳ Processing...</span>
      {/if}
    </div>
  {/if}

  <div class="stats-section">
    <div class="stat-card">
      <strong>{ENABLE_STREAMING ? 'Average WER' : 'Current WER'}</strong>
      <div class="stat-value">{wer || '—'}</div>
    </div>
    <div class="stat-card">
      <strong>Total Records</strong>
      <div class="stat-value">{records.length}</div>
    </div>
    <div class="stat-card">
      <strong>Reviewed</strong>
      <div class="stat-value">{Object.values(reviewed).filter(Boolean).length}</div>
    </div>
  </div>

  <div class="training-section">
    <button class="button wer-btn" on:click={calculateWEROnly} disabled={trainingInProgress}>
      {trainingInProgress && trainingStatus.message && trainingStatus.message.includes('WER') ? '⏳ Calculating...' : '📊 Calculate WER'}
    </button>
    <button class="button train-btn" on:click={triggerManualRetrain} disabled={trainingInProgress}>
      {trainingInProgress && !trainingStatus.message?.includes('WER') ? '⏳ Training...' : '🔄 Retrain Model'}
    </button>
    {#if trainingInProgress && trainingStatus.message}
      <small class="training-status">{trainingStatus.message}</small>
    {:else}
      <small>Calculate WER for benchmark, or retrain to improve the model</small>
    {#if ENABLE_STREAMING && streamingStats}
      <div class="stat-card">
        <strong>Total Corrections</strong>
        <div class="stat-value">{streamingStats.total_corrections || 0}</div>
      </div>
      <div class="stat-card">
        <strong>Avg Characters</strong>
        <div class="stat-value">{streamingStats.average_characters?.toFixed(0) || '—'}</div>
      </div>
    {:else}
      <div class="stat-card">
        <strong>Total Records</strong>
        <div class="stat-value">{records.length}</div>
      </div>
      <div class="stat-card">
        <strong>Reviewed</strong>
        <div class="stat-value">{Object.values(reviewed).filter(Boolean).length}</div>
      </div>
    {/if}
  </div>

  {#if ENABLE_TRAINING}
    <div class="training-section">
      <button class="button train-btn" on:click={triggerManualRetrain} disabled={trainingInProgress}>
        {trainingInProgress ? '⏳ Training...' : '🔄 Manual Retrain'}
      </button>
      {#if trainingInProgress && trainingStatus.message}
        <small class="training-status">{trainingStatus.message}</small>
      {:else}
        <small>Auto-retrains after 20 corrections</small>
      {/if}
    </div>
  {/if}
</div>

{#if ENABLE_TRAINING && trainingInProgress}
  <div class="training-progress-banner">
    <div class="spinner-small"></div>
    <div class="training-info">
      <strong>Training in Progress</strong>
      <span>{trainingStatus.message || 'Training the model...'}</span>
    </div>
    <div class="progress-bar-inline">
      <div class="progress-fill" style="width: {trainingStatus.progress || 0}%"></div>
    </div>
    <span class="progress-text">{trainingStatus.progress || 0}%</span>
  </div>
{/if}

{#if !ENABLE_STREAMING}
  {#if loading}
  <div class="loading-state">
    <div class="spinner"></div>
    <p>Loading records...</p>
  </div>
{:else if err}
  <div class="error-state">
    <p style="color:#e74c3c">❌ {err}</p>
    <button class="button" on:click={load}>Retry</button>
  </div>
{:else if records.length === 0}
  <div class="empty-state">
    <p>📭 No records yet. Upload or record audio to get started!</p>
  </div>
{:else}
  <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Time</th>
        <th>Audio</th>
        <th>Transcript</th>
        <th>Status</th>
        <th>Actions</th>
      </tr></thead>
      <tbody>
        {#each records as r,i (r.audio_file + i)}
          <tr class:not-reviewed-row={!reviewed[r.audio_file]} class:saving-row={saving[r.audio_file]}>
            <td class="time-col">{fmt(r.timestamp)}</td>
            <td class="small-audio">
              <audio controls src={`${BACKEND_URL}/audio/${r.audio_file}`}></audio>
            </td>
            <td style="width:100%;">
              <textarea
                rows="2"
                style="width:100%;box-sizing:border-box;"
                bind:value={manual[r.audio_file]}
                disabled={saving[r.audio_file]}
                placeholder="Edit transcription..."
              ></textarea>
              {#if hasUnsavedChanges(r.audio_file)}
                <small class="unsaved-indicator">⚠️ Unsaved changes</small>
              {:else if r.asr_transcript !== manual[r.audio_file]}
                <small class="edited-indicator">✏️ Edited</small>
              {/if}
            </td>
            <td style="text-align:center">
              {#if saving[r.audio_file]}
                <span class="saving-indicator">💾</span>
              {:else if hasUnsavedChanges(r.audio_file)}
                <button class="save-btn" on:click={()=>save(r)}>💾 Save</button>
              {:else if reviewed[r.audio_file]}
                <span class="checked">✔</span>
              {:else}
                <button class="approve-btn" title="Mark as correct" on:click={()=>approve(r)}>✓ Approve</button>
              {/if}
            </td>
            <td style="text-align:center">
              <button class="delete-btn" title="Delete this record" on:click={()=>deleteRecord(r)}>🗑️</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  {/if}
{/if}

{#if showNotif}
  <div class="snackbar {notifType}" transition:fly={{y:20}}>{notification}</div>
{/if}