const $ = id => document.getElementById(id);
const ctx = $('canvas').getContext('2d');
let socket, pending = false, stream, holistic, running = false, cameraFrames = [], generation = 0;
function status(message) { $('status').textContent = message; }
function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
  socket.onmessage = event => {
    pending = false;
    const data = JSON.parse(event.data);
    if (data.error) { status(data.error); return; }
    if (data.label !== undefined) {
      $('prediction').textContent = data.label || 'Uncertain';
      $('confidence').textContent = `${(data.confidence * 100).toFixed(1)}% model score · not calibrated confidence`;
      $('scores').replaceChildren();
      for (const [label, value] of Object.entries(data.scores)) {
        const row = document.createElement('div'); row.className = 'barrow';
        const line = document.createElement('div');
        const name = document.createElement('span'); name.textContent = label;
        const score = document.createElement('span'); score.textContent = `${(value * 100).toFixed(1)}%`;
        line.append(name, score);
        const bar = document.createElement('div'); bar.className = 'bar';
        const fill = document.createElement('i'); fill.style.width = `${value * 100}%`;
        bar.append(fill); row.append(line, bar); $('scores').append(row);
      }
      status('Inference complete');
    }
    if (data.tokens) $('tokens').textContent = data.tokens.join(' · ') || 'Your motion sequence appears here.';
  };
  socket.onclose = () => { pending = false; status('Disconnected. Reload to reconnect.'); };
  socket.onerror = () => status('Local server connection failed.');
}
function send(frames) {
  if (socket.readyState !== WebSocket.OPEN || pending) return;
  pending = true; socket.send(JSON.stringify({frames}));
}
function draw(points, trail = []) {
  ctx.clearRect(0, 0, 900, 540);
  ctx.strokeStyle = '#21343f'; ctx.lineWidth = 1;
  for(let x=0;x<900;x+=45){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,540);ctx.stroke();}
  for(let y=0;y<540;y+=45){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(900,y);ctx.stroke();}
  ctx.strokeStyle='#5b8b82';ctx.lineWidth=2;ctx.beginPath();
  trail.forEach((p,i)=>i?ctx.lineTo(p[0]*900,p[1]*540):ctx.moveTo(p[0]*900,p[1]*540));ctx.stroke();
  for(const [i,p] of points.entries()) {
    if(p.every(v=>v===0))continue;
    ctx.fillStyle=i>=501?'#adf5cd':i<33?'#88b8de':'#456573';
    ctx.beginPath();ctx.arc(p[0]*900,p[1]*540,i>=501?3:i<33?5:1,0,Math.PI*2);ctx.fill();
  }
  if(points[11]&&points[12]){ctx.strokeStyle='#88b8de';ctx.beginPath();ctx.moveTo(points[11][0]*900,points[11][1]*540);ctx.lineTo(points[12][0]*900,points[12][1]*540);ctx.stroke();}
}
function stop() {
  generation++; running=false; cameraFrames=[];
  if(stream)stream.getTracks().forEach(track=>track.stop());
  stream=null; $('video').srcObject=null; $('video').hidden=true;
  $('stop').disabled=true; $('camera').disabled=false; $('play').disabled=false;
  status('Stopped');
}
$('play').onclick = async () => {
  stop(); const epoch = generation; $('play').disabled=true;
  $('source').textContent='SYNTHETIC FIXTURE';
  try {
    const response=await fetch(`/api/fixture/${$('motion').value}?seed=${Math.floor(Math.random()*1000000)}`);
    if(!response.ok)throw Error('Fixture unavailable');
    const {frames}=await response.json(); const trail=[];
    for(const frame of frames){if(epoch!==generation)return;trail.push(frame[522]);draw(frame,trail);status('Playing procedural landmarks');await new Promise(r=>setTimeout(r,45));}
    send(frames);
  } catch(error){status(error.message);} finally{$('play').disabled=false;}
};
function loadScript(url) { return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=url;s.onload=resolve;s.onerror=()=>reject(Error('MediaPipe download failed. Use the offline fixture demo.'));document.head.append(s);}); }
$('camera').onclick = async () => {
  stop(); $('camera').disabled=true; $('play').disabled=true; $('stop').disabled=false;
  const epoch=generation;
  try {
    status('Loading MediaPipe…');
    if(!window.Holistic)await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.js');
    if(epoch!==generation)return;
    if(!holistic){holistic=new Holistic({locateFile:file=>`https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`});
      holistic.setOptions({modelComplexity:0,smoothLandmarks:true,refineFaceLandmarks:false,minDetectionConfidence:0.5,minTrackingConfidence:0.5});
      holistic.onResults(result=>{
        if(!running)return;
        const pack=(items,n)=>Array.from({length:n},(_,i)=>items?.[i]?[items[i].x,items[i].y,items[i].z]:[0,0,0]);
        const frame=[...pack(result.poseLandmarks,33),...pack(result.faceLandmarks,468),...pack(result.leftHandLandmarks,21),...pack(result.rightHandLandmarks,21)];
        draw(frame);cameraFrames.push(frame);
        if(cameraFrames.length===32){send(cameraFrames);cameraFrames=[];}
      });}
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    if(epoch!==generation){stop();return;}
    $('video').srcObject=stream;$('video').hidden=false;await $('video').play();running=true;
    $('source').textContent='CAMERA · EXPERIMENTAL';status('Tracking locally');
    while(running&&epoch===generation){await holistic.send({image:$('video')});await new Promise(r=>setTimeout(r,65));}
  }catch(error){stop();status(error.message);}
};
$('stop').onclick=stop;
$('clear').onclick=()=>{if(socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({reset:true}));};
window.addEventListener('pagehide',()=>{stop();socket?.close();});
fetch('/api/info').then(r=>r.json()).then(data=>{$('evaluation').textContent=`${data.evaluation.test*1} synthetic test clips`;}).catch(()=>status('Could not load model metadata'));
draw([]);connect();
