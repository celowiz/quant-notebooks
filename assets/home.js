(function(){
const {preview,rnd,walk}=window.QNPreview;
// ---- Hero: partículas sutis + curva de preço
const cv=document.getElementById('hero-bg');
if(cv){const ctx=cv.getContext('2d');const r=rnd(3);const pts=[];for(let i=0;i<80;i++)pts.push({x:r(),y:r()*.9,rad:r()*1.5+.4,a:r()*.28+.04,vx:(r()-.5)*.00012,vy:(r()-.5)*.00008});
 const series=walk(180,r,3);const mn=Math.min(...series),mx=Math.max(...series);
 const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;let W,H,dpr,prog=reduce?1:0,t0=null;
 function size(){dpr=Math.min(devicePixelRatio||1,2);W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}
 function curve(){const top=H*.42,bot=H*.92,x1=W*.04,x2=W*.80;return series.map((v,i)=>[x1+i*(x2-x1)/(series.length-1),bot-(v-mn)/(mx-mn)*(bot-top)])}
 function frame(ts){if(t0==null)t0=ts;if(!reduce)prog=Math.min(1,(ts-t0)/2200);
  ctx.clearRect(0,0,W,H);ctx.strokeStyle='rgba(255,255,255,.035)';ctx.lineWidth=1;for(let i=1;i<7;i++){const y=i*H/7;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  for(const p of pts){if(!reduce){p.x=(p.x+p.vx+1)%1;p.y=(p.y+p.vy+1)%1}ctx.fillStyle=`rgba(244,241,234,${p.a})`;ctx.beginPath();ctx.arc(p.x*W,p.y*H,p.rad,0,7);ctx.fill()}
  const c=curve(),n=Math.max(2,Math.floor(c.length*prog));const seg=c.slice(0,n);
  const g=ctx.createLinearGradient(0,H*.42,0,H);g.addColorStop(0,'rgba(45,212,191,.22)');g.addColorStop(1,'rgba(45,212,191,0)');
  ctx.beginPath();seg.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.lineTo(seg[n-1][0],H);ctx.lineTo(seg[0][0],H);ctx.closePath();ctx.fillStyle=g;ctx.fill();
  ctx.beginPath();seg.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle='#2dd4bf';ctx.lineWidth=2.4;ctx.shadowColor='rgba(45,212,191,.6)';ctx.shadowBlur=14;ctx.stroke();ctx.shadowBlur=0;
  const L=seg[n-1];const pulse=reduce?0:(Math.sin(ts/500)+1)/2;ctx.fillStyle=`rgba(45,212,191,${.15+pulse*.15})`;ctx.beginPath();ctx.arc(L[0],L[1],10+pulse*6,0,7);ctx.fill();ctx.fillStyle='#2dd4bf';ctx.beginPath();ctx.arc(L[0],L[1],4.5,0,7);ctx.fill();
  if(!reduce)requestAnimationFrame(frame)}
 size();addEventListener('resize',()=>{size();if(reduce)frame(0)});requestAnimationFrame(frame)}
// ---- Destaque
const f=document.getElementById('feat-svg');if(f)f.innerHTML=preview('rrg',0,426,230);
// ---- Miniaturas + filtros
document.querySelectorAll('.card svg[data-kind]').forEach((el,k)=>el.innerHTML=preview(el.dataset.kind,k,392,150));
const state={theme:'Todos',level:null,q:''};
function apply(){let vis=0;document.querySelectorAll('.card').forEach(c=>{const ok=(state.theme=='Todos'||c.dataset.theme==state.theme)&&(!state.level||c.dataset.level==state.level)&&(!state.q||c.dataset.search.includes(state.q));c.style.display=ok?'':'none';vis+=ok});document.getElementById('empty').style.display=vis?'none':'block'}
document.querySelectorAll('[data-theme-f]').forEach(b=>b.onclick=()=>{state.theme=b.dataset.themeF;document.querySelectorAll('[data-theme-f]').forEach(x=>x.classList.toggle('on',x==b));apply()});
document.querySelectorAll('[data-level-f]').forEach(b=>b.onclick=()=>{state.level=state.level==b.dataset.levelF?null:b.dataset.levelF;document.querySelectorAll('[data-level-f]').forEach(x=>x.classList.toggle('on',x.dataset.levelF==state.level));apply()});
const s=document.getElementById('q');if(s)s.oninput=()=>{state.q=s.value.trim().toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');apply()};
})();
