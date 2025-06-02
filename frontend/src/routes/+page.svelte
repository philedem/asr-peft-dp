<script lang="ts">
  import { onMount } from 'svelte';
  import { fly }     from 'svelte/transition';

  let records:any[] = [];
  let loading=false, err='';
  let notification='', notifType='success', showNotif=false;
  let manual:Record<string,string>={}, reviewed:Record<string,boolean>={}, saving:Record<string,boolean>={};
  let wer='';
  let isRecording=false, chunks:Blob[]=[], recorder:MediaRecorder|null=null;

  const toast=(m:string,t='success')=>{
    notification=m; notifType=t; showNotif=true;
    setTimeout(()=>showNotif=false,2300);
  };

  // ------- API helpers ------------
  const getJSON = (u:string)=>fetch(u).then(r=>r.json());
  const post    = (u:string,body:any)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});

  async function fetchWER(){ wer=(await getJSON('http://localhost:8000/asr/wer')).wer||''; }
  async function load(){
    loading=true; err='';
    try{
      records=await getJSON('http://localhost:8000/asr/records');
      records.sort((a,b)=>(b.timestamp??'').localeCompare(a.timestamp??''));
      manual={}; reviewed={};
      records.forEach(r=>{
        manual[r.audio_file] = r.manual_transcript?.length ? r.manual_transcript : r.asr_transcript || '';
        reviewed[r.audio_file] = !!(r.manual_transcript && r.manual_transcript.length > 0);
      });
    }catch(e){err='could not load';}
    loading=false;
  }

  async function save(r){
    saving[r.audio_file]=true;
    await post('http://localhost:8000/asr/save_record',{
      audio_id:r.audio_file,
      asr_transcript:r.asr_transcript,
      manual_transcript:manual[r.audio_file],
      flagged:r.flagged
    });
    reviewed[r.audio_file] = !!(r.manual_transcript && r.manual_transcript.length > 0);
    toast('saved'); await load(); await fetchWER();
    saving[r.audio_file]=false;
  }

  // ---------- recording ----------
  async function toggleRec(){
    if(!isRecording){
      chunks=[]; isRecording=true;
      const stream=await navigator.mediaDevices.getUserMedia({audio:true});
      recorder=new MediaRecorder(stream);
      recorder.ondataavailable=e=>chunks.push(e.data);
      recorder.onstop=async ()=>{
        isRecording=false;
        await sendToASR(new Blob(chunks,{type:'audio/webm'}));
      };
      recorder.start();
    }else recorder?.stop();
  }
  async function sendToASR(blob:Blob){
    const fd=new FormData(); fd.append('audio',blob,'record.webm');
    toast('sending...');
    const r=await fetch('http://localhost:8000/asr/transcribe',{method:'POST',body:fd});
    r.ok?toast('done'):toast('ASR fail','error');
    await load(); await fetchWER();
  }
  const approve=r=>{ manual[r.audio_file]=r.asr_transcript; save(r); };

  onMount(()=>{ load(); fetchWER(); setInterval(fetchWER,70000); });

  const fmt=t=>t?.replace('T',' ').replace('Z','').slice(0,19);
</script>

<h1>ASR Annotation Table</h1>

<div class="audio-section">
  <button class="button" on:click={toggleRec}>{isRecording?'Stop':'Start'} Recording</button>
  <input type="file" accept="audio/*" on:change={e=>sendToASR(e.target.files[0])}/>
</div>

<div class="current-wer">
  <strong>Current WER:</strong> {wer||'—'}
</div>

{#if loading}
  <p>Loading records...</p>
{:else if err}
  <p style="color:#e74c3c">{err}</p>
{:else}
  <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Time</th><th>Audio</th><th>Transcript</th><th></th>
      </tr></thead>
      <tbody>
        {#each records as r,i (r.audio_file + i)}
          <tr class:not-reviewed-row={!reviewed[r.audio_file]}>
            <td>{fmt(r.timestamp)}</td>
            <td class="small-audio"><audio controls src={`http://localhost:8000/audio/${r.audio_file}`}></audio></td>
            <td style="width:100%;">
              <textarea
                rows="2"
                style="width:100%;box-sizing:border-box;"
                bind:value={manual[r.audio_file]}
                on:blur={()=>save(r)}
                disabled={saving[r.audio_file]}>
              </textarea>
            </td>
            <td style="text-align:center">
              {#if reviewed[r.audio_file]}
                ✔
              {:else}
                <button class="approve-btn" title="Mark correct" on:click={()=>approve(r)}>✓</button>
              {/if}
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