<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fly }     from 'svelte/transition';

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const WER_POLL_INTERVAL = import.meta.env.VITE_WER_POLL_INTERVAL || 70000;
  const AUDIO_INPUTS = (import.meta.env.VITE_AUDIO_INPUTS || 'mic,upload,rtsp').split(',').map((s:string)=>s.trim());

  let records:any[] = [];
  let loading=false, err='';
  let notification='', notifType='success', showNotif=false;
  let manual:Record<string,string>={}, reviewed:Record<string,boolean>={}, saving:Record<string,boolean>={};
  let originalManual:Record<string,string>={};
  let wer='N/A';
  let modelInfo:any = null;
  let trainingStatus:any = {status: 'idle', progress: 0, message: ''};
  let isRecording=false;
  let uploadProgress=false;
  // Live VAD recording state
  let micStream:MediaStream|null=null;
  let audioCtx:AudioContext|null=null;
  let micProcessor:ScriptProcessorNode|null=null;
  let pcmBuffer:Float32Array[]=[];
  let pcmSamples=0;
  let silentWindows=0;
  let micSegments=0;
  let sendingSegment=false;
  const MIC_SR = 16000;
  const VAD_WINDOW = 0.1;          // 100ms windows
  const SILENCE_THRESH = 0.02;     // RMS threshold (stricter to avoid sending silence)
  const SILENCE_DUR = 1.5;         // seconds of silence to trigger
  const MIN_SPEECH = 1.5;          // min segment length (seconds)
  const MAX_SEGMENT = 15;          // max segment length (seconds)
  let trainingInProgress=false;
  let audioDevices:MediaDeviceInfo[]=[];
  let selectedDeviceId:string='';

  // RTSP streaming state
  let rtspUrl:string = '';
  let isStreaming=false;
  let streamStatus:any = {is_running: false, segments_created: 0, elapsed_time: 0};
  let streamPollTimer:number|null=null;
  let showRtspInput=false;

  const toast=(m:string,t='success')=>{
    notification=m; notifType=t; showNotif=true;
    setTimeout(()=>showNotif=false,3000);
  };

  // ------- API helpers ------------
  const getJSON = async (u:string)=>{
    const response = await fetch(u);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  };
  const postJSON = async (u:string,body:any)=>{
    const response = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  };

  async function fetchWER(){ 
    try {
      const data = await getJSON(`${BACKEND_URL}/asr/wer`);
      wer = data.wer || 'N/A';
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
      if (wasTraining && status.status === 'completed') {
        toast('Training completed successfully!', 'success');
        await fetchWER(); await fetchModelInfo(); await load();
      } else if (wasTraining && status.status === 'failed') {
        toast('Training failed', 'error');
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
        originalManual[r.audio_file] = manual[r.audio_file];
        reviewed[r.audio_file] = !!(r.manual_transcript && r.manual_transcript.length > 0);
      });
    }catch(e){
      err='Could not load records. Is the backend running?';
      console.error(e);
    }
    loading=false;
  }

  // Fetch only new records and prepend — does not disturb existing rows or audio playback
  async function loadNew(){
    try{
      const fresh = await getJSON(`${BACKEND_URL}/asr/records`);
      const known = new Set(records.map((r:any)=>r.audio_file));
      const added = fresh.filter((r:any)=>!known.has(r.audio_file));
      if(added.length === 0) return;
      added.sort((a:any,b:any)=>(b.timestamp??'').localeCompare(a.timestamp??''));
      added.forEach((r:any)=>{
        manual[r.audio_file] = r.manual_transcript?.length ? r.manual_transcript : r.asr_transcript || '';
        originalManual[r.audio_file] = manual[r.audio_file];
        reviewed[r.audio_file] = !!(r.manual_transcript && r.manual_transcript.length > 0);
      });
      records = [...added, ...records];
    }catch(e){
      console.error('loadNew failed:', e);
    }
  }

  async function save(r:any, force=false){
    if (saving[r.audio_file]) return;
    if (!force && manual[r.audio_file] === originalManual[r.audio_file]) return;
    saving[r.audio_file]=true;
    try {
      await postJSON(`${BACKEND_URL}/asr/save_record`,{
        audio_id:r.audio_file,
        asr_transcript:r.asr_transcript,
        manual_transcript:manual[r.audio_file],
      });
      reviewed[r.audio_file] = !!(manual[r.audio_file] && manual[r.audio_file].length > 0);
      originalManual[r.audio_file] = manual[r.audio_file];
      toast('Saved successfully'); 
      await load(); await fetchWER();
    } catch(e) {
      toast('Save failed', 'error');
    } finally {
      saving[r.audio_file]=false;
    }
  }
  
  const hasUnsavedChanges = (audioFile:string) => manual[audioFile] !== originalManual[audioFile];

  async function deleteRecord(r:any) {
    if (!confirm(`Delete this record?\n\nAudio: ${r.audio_file}`)) return;
    try {
      const res = await fetch(`${BACKEND_URL}/asr/records/${r.audio_file.replace('.wav', '')}`, {method: 'DELETE'});
      if (!res.ok) throw new Error('Delete failed');
      toast('Record deleted');
      await load();
    } catch(e) {
      toast('Delete failed', 'error');
    }
  }

  // ---------- audio device enumeration ----------
  async function loadAudioDevices(){
    try {
      const tempStream=await navigator.mediaDevices.getUserMedia({audio:true});
      tempStream.getTracks().forEach(t=>t.stop());
      const devices=await navigator.mediaDevices.enumerateDevices();
      audioDevices=devices.filter(d=>d.kind==='audioinput');
      if(!selectedDeviceId || !audioDevices.find(d=>d.deviceId===selectedDeviceId)){
        selectedDeviceId=audioDevices[0]?.deviceId||'';
      }
    } catch(e){
      console.error('Could not enumerate audio devices:',e);
    }
  }

  // ---------- recording with live VAD ----------
  function rms(buf:Float32Array):number {
    let sum=0;
    for(let i=0;i<buf.length;i++) sum+=buf[i]*buf[i];
    return Math.sqrt(sum/buf.length);
  }

  function buildWavBlob(samples:Float32Array):Blob {
    const len=samples.length;
    const buf=new ArrayBuffer(44+len*2);
    const v=new DataView(buf);
    const writeStr=(o:number,s:string)=>{for(let i=0;i<s.length;i++) v.setUint8(o+i,s.charCodeAt(i));};
    writeStr(0,'RIFF'); v.setUint32(4,36+len*2,true); writeStr(8,'WAVE');
    writeStr(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true);
    v.setUint16(22,1,true); v.setUint32(24,MIC_SR,true);
    v.setUint32(28,MIC_SR*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
    writeStr(36,'data'); v.setUint32(40,len*2,true);
    for(let i=0;i<len;i++){
      const s=Math.max(-1,Math.min(1,samples[i]));
      v.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
    }
    return new Blob([buf],{type:'audio/wav'});
  }

  let speechDetected=false; // tracks if any speech was seen in current buffer

  async function flushSegment(){
    if(sendingSegment || pcmBuffer.length===0) return;
    const totalSamples=pcmBuffer.reduce((a,b)=>a+b.length,0);
    if(totalSamples < MIC_SR * MIN_SPEECH) return; // too short

    // Check overall RMS — skip if it's all silence
    if(!speechDetected){
      pcmBuffer=[]; pcmSamples=0; silentWindows=0;
      return;
    }

    sendingSegment=true;
    // Concatenate buffer
    const full=new Float32Array(totalSamples);
    let off=0;
    for(const chunk of pcmBuffer){ full.set(chunk,off); off+=chunk.length; }
    pcmBuffer=[]; pcmSamples=0; silentWindows=0; speechDetected=false;

    // Build WAV and send
    const wav=buildWavBlob(full);
    try {
      const fd=new FormData();
      fd.append('audio',wav,`mic_${Date.now()}.wav`);
      const r=await fetch(`${BACKEND_URL}/asr/transcribe`,{method:'POST',body:fd});
      if(r.ok){
        const result = await r.json();
        if(result.created > 0){
          micSegments += result.created;
          await loadNew();
        }
      }
    } catch(e){
      console.error('Failed to send segment:',e);
    } finally {
      sendingSegment=false;
    }
  }

  async function toggleRec(){
    if(!isRecording){
      try {
        isRecording=true; pcmBuffer=[]; pcmSamples=0; silentWindows=0; micSegments=0; speechDetected=false;
        const audioConstraints:any = {
          channelCount:1, sampleRate:MIC_SR,
          echoCancellation:true, noiseSuppression:true
        };
        if(selectedDeviceId) audioConstraints.deviceId={exact:selectedDeviceId};
        micStream=await navigator.mediaDevices.getUserMedia({audio:audioConstraints});
        audioCtx=new AudioContext({sampleRate:MIC_SR});
        const source=audioCtx.createMediaStreamSource(micStream);
        // 4096 samples at 16kHz ≈ 256ms per callback
        micProcessor=audioCtx.createScriptProcessor(4096,1,1);

        micProcessor.onaudioprocess=(e)=>{
          if(!isRecording) return;
          const input=e.inputBuffer.getChannelData(0);
          const copy=new Float32Array(input);
          pcmBuffer.push(copy);
          pcmSamples+=copy.length;

          // VAD: check trailing silence
          if(rms(copy)<SILENCE_THRESH){
            silentWindows++;
          } else {
            silentWindows=0;
            speechDetected=true;
          }

          const elapsed=pcmSamples/MIC_SR;
          const silenceSec=silentWindows*(4096/MIC_SR);

          // Trigger: enough silence after some speech, or max duration
          if((elapsed>=MIN_SPEECH+SILENCE_DUR && silenceSec>=SILENCE_DUR) || elapsed>=MAX_SEGMENT){
            flushSegment();
          }
        };

        source.connect(micProcessor);
        micProcessor.connect(audioCtx.destination);
        toast('Listening...');
      } catch(e) {
        toast('Microphone access denied', 'error');
        isRecording=false;
      }
    } else {
      isRecording=false;
      // Flush any remaining audio
      await flushSegment();
      // Cleanup
      micProcessor?.disconnect();
      micProcessor=null;
      if(micStream){ micStream.getTracks().forEach(t=>t.stop()); micStream=null; }
      if(audioCtx){ audioCtx.close(); audioCtx=null; }
      toast(`Stopped. ${micSegments} segment(s) transcribed.`);
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
        toast(`Created ${result.created} transcription(s)`);
      } else {
        const errorData = await r.json().catch(() => ({ error: 'Unknown error' }));
        toast(`Transcription failed: ${errorData.error}`, 'error');
      }
      await load(); await fetchWER();
    } catch(e) {
      toast('Upload/transcription failed', 'error');
      console.error(e);
    } finally {
      uploadProgress=false;
    }
  }
  
  const approve=async (r:any)=>{ 
    manual[r.audio_file]=r.asr_transcript; 
    await save(r, true);
  };

  // ---------- RTSP streaming ----------
  async function toggleStream(){
    if(isStreaming){
      await stopStream();
    } else {
      await startStream();
    }
  }

  async function startStream(){
    if(!rtspUrl.trim()){ toast('Enter an RTSP URL','error'); return; }
    try {
      await postJSON(`${BACKEND_URL}/stream/start`, {
        rtsp_url: rtspUrl,
        segment_duration: 10,
        silence_duration: 2.0,
        silence_threshold: 0.02,
      });
      isStreaming=true;
      toast('RTSP streaming started');
      let lastSegCount = 0;
      streamPollTimer = setInterval(async ()=>{
        try {
          streamStatus = await getJSON(`${BACKEND_URL}/stream/status`);
          if(!streamStatus.is_running && isStreaming){
            isStreaming=false;
            clearInterval(streamPollTimer!);
            streamPollTimer=null;
            if(streamStatus.error) toast(`Stream error: ${streamStatus.error}`,'error');
            else toast(`Stream ended. ${streamStatus.segments_created} segments created.`);
          }
          if(streamStatus.segments_created > lastSegCount){
            lastSegCount = streamStatus.segments_created;
            await loadNew();
          }
        } catch(e){ console.error('Stream poll error:', e); }
      }, 3000);
    } catch(e:any) {
      toast(`Failed to start stream: ${e.message}`,'error');
    }
  }

  async function stopStream(){
    try {
      const res = await postJSON(`${BACKEND_URL}/stream/stop`, {});
      isStreaming=false;
      if(streamPollTimer){ clearInterval(streamPollTimer); streamPollTimer=null; }
      toast(`Stream stopped. ${res.segments_created} segments created.`);
      await load();
    } catch(e:any) {
      toast(`Failed to stop stream: ${e.message}`,'error');
    }
  }

  // ---------- training ----------
  async function triggerManualRetrain() {
    if (trainingInProgress) return;
    trainingInProgress = true;
    toast('Starting training...', 'success');
    try {
      await getJSON(`${BACKEND_URL}/train/retrain_lora`);
    } catch(e) {
      toast('Failed to start training', 'error');
      trainingInProgress = false;
    }
  }

  async function calculateWEROnly() {
    if (trainingInProgress) return;
    const reviewedCount = Object.values(reviewed).filter(Boolean).length;
    if (reviewedCount === 0) {
      toast('No reviewed records to calculate WER.', 'error');
      return;
    }
    trainingInProgress = true;
    toast('Calculating WER...', 'success');
    try {
      await getJSON(`${BACKEND_URL}/train/calculate_wer`);
      const pollCompletion = async () => {
        for (let i = 0; i < 20; i++) {
          await new Promise(resolve => setTimeout(resolve, 500));
          const status = await getJSON(`${BACKEND_URL}/train/status`);
          if (status.status === 'completed') {
            trainingInProgress = false;
            await fetchWER();
            toast('WER calculation completed!', 'success');
            return;
          } else if (status.status === 'failed') {
            trainingInProgress = false;
            toast('WER calculation failed', 'error');
            return;
          }
        }
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
    fetchWER();
    fetchModelInfo();
    fetchTrainingStatus();
    setInterval(fetchWER, WER_POLL_INTERVAL);
    setInterval(fetchModelInfo, WER_POLL_INTERVAL);
    setInterval(fetchTrainingStatus, 5000);
    // Check if a stream is already running
    getJSON(`${BACKEND_URL}/stream/status`).then(s => {
      if(s.is_running){
        isStreaming=true; streamStatus=s; rtspUrl=s.rtsp_url; showRtspInput=true;
        let lastSegCount = s.segments_created;
        streamPollTimer = setInterval(async ()=>{
          try {
            streamStatus = await getJSON(`${BACKEND_URL}/stream/status`);
            if(!streamStatus.is_running){ isStreaming=false; clearInterval(streamPollTimer!); streamPollTimer=null; }
            if(streamStatus.segments_created > lastSegCount){
              lastSegCount = streamStatus.segments_created;
              await loadNew();
            }
          } catch(e){}
        }, 3000);
      }
    }).catch(()=>{});
  });

  onDestroy(() => {
    if(streamPollTimer) clearInterval(streamPollTimer);
  });

  const fmt=(t:any)=>t?.replace('T',' ').replace('Z','').slice(0,19);
</script>

<h1>ASR Annotation System</h1>
<p class="subtitle">Transcription & Model Training</p>

{#if modelInfo}
  <div class="model-info-banner">
    <span class="model-name">Model: <strong>{modelInfo.base_model}</strong></span>
    <span class="model-status">
      {#if modelInfo.has_lora_adapter}
        LoRA Fine-tuned {modelInfo.training_iteration ? `(Iteration #${modelInfo.training_iteration})` : ''}
      {:else}
        Base Model (No Fine-tuning)
      {/if}
    </span>
    <span class="device-info">{modelInfo.device.toUpperCase()}</span>
  </div>
{/if}

<div class="control-panel">
  <!-- Audio Input: Mic + Upload + RTSP -->
  <div class="audio-section">
    <h3>Audio Input</h3>
    {#if AUDIO_INPUTS.includes('mic')}
    {#if audioDevices.length > 0}
      <select class="device-select" bind:value={selectedDeviceId} disabled={isRecording || uploadProgress}>
        {#each audioDevices as dev}
          <option value={dev.deviceId}>{dev.label || `Microphone ${audioDevices.indexOf(dev)+1}`}</option>
        {/each}
      </select>
    {/if}
    <button class="button record-btn" class:recording={isRecording} on:click={toggleRec} disabled={uploadProgress || isStreaming}>
      {isRecording ? '⏹ Stop' : '🎤 Record'}
    </button>
    {#if isRecording}
      <span class="mic-live-indicator">
        <span class="pulse-dot"></span>
        Listening — {micSegments} segment{micSegments!==1?'s':''}
      </span>
    {/if}
    {/if}
    {#if AUDIO_INPUTS.includes('upload')}
    <label class="button upload-btn">
      📁 Upload Audio
      <input type="file" accept="audio/*" style="display:none" on:change={e => { const t = e.target as HTMLInputElement; if(t.files?.[0]) sendToASR(t.files[0]); }} disabled={uploadProgress}/>
    </label>
    {/if}

    {#if AUDIO_INPUTS.includes('rtsp')}
    <button class="button rtsp-toggle-btn" on:click={()=>showRtspInput=!showRtspInput}>
      {showRtspInput ? '▾ RTSP Stream' : '▸ RTSP Stream'}
    </button>
    {#if showRtspInput}
      <div class="rtsp-input-group">
        <input
          type="text"
          class="rtsp-url-input"
          bind:value={rtspUrl}
          placeholder="rtsp://user:pass@host:554/stream"
          disabled={isStreaming}
        />
        <button class="button stream-btn" class:streaming={isStreaming} on:click={toggleStream} disabled={!rtspUrl.trim() && !isStreaming}>
          {isStreaming ? '⏹ Stop' : '▶ Start'}
        </button>
      </div>
      {#if isStreaming}
        <div class="stream-active-indicator">
          <span class="pulse-dot"></span>
          Streaming — {streamStatus.segments_created} segments — {Math.floor(streamStatus.elapsed_time)}s
        </div>
      {/if}
    {/if}
    {/if}

    {#if uploadProgress}
      <span class="status-indicator">⏳ Processing...</span>
    {/if}
  </div>

  <!-- Stats -->
  <div class="stats-section">
    <div class="stat-card">
      <strong>Current WER</strong>
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

  <!-- Training -->
  <div class="training-section">
    <button class="button wer-btn" on:click={calculateWEROnly} disabled={trainingInProgress}>
      {trainingInProgress && trainingStatus.message?.includes('WER') ? '⏳ Calculating...' : '📊 Calculate WER'}
    </button>
    <button class="button train-btn" on:click={triggerManualRetrain} disabled={trainingInProgress}>
      {trainingInProgress && !trainingStatus.message?.includes('WER') ? '⏳ Training...' : '🔄 Retrain Model'}
    </button>
    {#if trainingInProgress && trainingStatus.message}
      <small class="training-status">{trainingStatus.message}</small>
    {:else}
      <small>Calculate WER for benchmark, or retrain to improve the model</small>
    {/if}
  </div>
</div>

{#if trainingInProgress}
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

{#if loading}
  <div class="loading-state">
    <div class="spinner"></div>
    <p>Loading records...</p>
  </div>
{:else if err}
  <div class="error-state">
    <p style="color:var(--error)">❌ {err}</p>
    <button class="button" on:click={load}>Retry</button>
  </div>
{:else if records.length === 0}
  <div class="empty-state">
    <p>No records yet. Upload or record audio to get started!</p>
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

{#if showNotif}
  <div class="snackbar {notifType}" transition:fly={{y:20}}>{notification}</div>
{/if}
