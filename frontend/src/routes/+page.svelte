<script lang="ts">
  import { onMount } from 'svelte';
  import { fly }     from 'svelte/transition';

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const WER_POLL_INTERVAL = import.meta.env.VITE_WER_POLL_INTERVAL || 70000;

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

  // ---------- recording ----------
  async function toggleRec(){
    if(!isRecording){
      try {
        chunks=[]; isRecording=true;
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
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

  onMount(()=>{ 
    load(); 
    fetchWER();
    fetchModelInfo();
    fetchTrainingStatus();
    setInterval(fetchWER, WER_POLL_INTERVAL);
    setInterval(fetchModelInfo, WER_POLL_INTERVAL); // Poll model info alongside WER
    setInterval(fetchTrainingStatus, 5000); // Poll training status more frequently (every 5s)
  });

  const fmt=t=>t?.replace('T',' ').replace('Z','').slice(0,19);
</script>

<h1>ASR Annotation System</h1>
<p class="subtitle">Transcription & Model Training</p>

{#if modelInfo}
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

{#if showNotif}
  <div class="snackbar {notifType}" transition:fly={{y:20}}>{notification}</div>
{/if}