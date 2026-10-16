// Miniaturas ilustrativas (dados sintéticos, apenas decorativos)
(function(){
const T='#2dd4bf';
function rnd(s){return()=>(s=Math.sin(s*9301+49297)*233280,s-Math.floor(s))}
function walk(n,r,v){let y=0,a=[];for(let i=0;i<n;i++){y+=(r()-.47)*(v||1);a.push(y)}return a}
function path(a,w,h,p){p=p||6;const mn=Math.min(...a),mx=Math.max(...a);return a.map((y,i)=>(i?'L':'M')+(p+i*(w-2*p)/(a.length-1)).toFixed(1)+','+(h-p-(y-mn)/(mx-mn||1)*(h-2*p)).toFixed(1)).join('')}
function rrg(w,h,big){const cx=w/2,cy=h/2;let s=`<rect x=${cx} y=0 width=${cx} height=${cy} fill="#22c55e" opacity=.06 /><rect x=${cx} y=${cy} width=${cx} height=${cy} fill="#eab308" opacity=.06 /><rect x=0 y=${cy} width=${cx} height=${cy} fill="#ef4444" opacity=.06 /><rect x=0 y=0 width=${cx} height=${cy} fill="#3b82f6" opacity=.06 /><line x1=${cx} x2=${cx} y1=0 y2=${h} stroke="#3f3f46"/><line y1=${cy} y2=${cy} x1=0 x2=${w} stroke="#3f3f46"/>`;
const fs=big?12:9;[['Liderando',w-6,14,'end','#4ade80'],['Enfraquecendo',w-6,h-6,'end','#facc15'],['Atrasado',6,h-6,'start','#f87171'],['Melhorando',6,14,'start','#60a5fa']].forEach(q=>s+=`<text x=${q[1]} y=${q[2]} text-anchor=${q[3]} fill="${q[4]}" font-size=${fs} font-family="ui-monospace,monospace" opacity=.8>${q[0]}</text>`);
const names=['FIN','MAT','ENE','CON','UTL','IMOB'],cols=['#2dd4bf','#f4f1ea','#fb923c','#a78bfa','#60a5fa','#f472b6'];
names.forEach((n,k)=>{const ex=[.55,.3,-.45,-.5,.4,-.2][k],ey=[.4,-.45,-.35,.4,.1,-.6][k],ph=k*.9;let pts=[];for(let i=0;i<8;i++){const t=ph+i*.28;pts.push([(cx+(ex*.75+Math.cos(t)*.18)*cx*(.4+i*.085)).toFixed(1),(cy-(ey*.75+Math.sin(t)*.18)*cy*(.4+i*.085)).toFixed(1)])}
s+=`<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${cols[k]}" stroke-width="${big?2:1.4}" opacity=".85"/>`;pts.forEach((p,i)=>{s+=`<circle cx="${p[0]}" cy="${p[1]}" r="${i==7?(big?5:4):1.8}" fill="${cols[k]}" opacity="${.3+i*.1}"/>`});const L=pts[7];s+=`<text x="${+L[0]+8}" y="${+L[1]+4}" fill="${cols[k]}" font-size="${big?12:10}" font-family="ui-monospace,monospace">${n}</text>`});return s}
function preview(kind,seed,W,H){let r=rnd(seed+5),s='';
if(kind=='rrg')return rrg(W,H,W>400);
if(kind=='bars'){[22,14,11,9,8,7,6,5,4,3,3,2].forEach((x,i)=>s+=`<rect x=${16+i*31} y=${H-12-x*5} width=22 height=${x*5} rx=2 fill="${i?'#3f3f46':T}"/>`)}
else if(kind=='multi'){['#2dd4bf','#f4f1ea','#71717a'].forEach((c,j)=>s+=`<path d="${path(walk(60,r,1),W,H,14)}" stroke="${c}" fill=none stroke-width=1.5 opacity=${1-j*.25} />`)}
else if(kind=='gov'){[1,-.5,1.6,.4,-1.2,.9,1.3].forEach((x,i)=>{const y=H/2,hh=x*35;s+=`<rect x=${24+i*50} y=${hh>0?y-hh:y} width=34 height=${Math.abs(hh)} fill="${hh>0?T:'#ef4444'}" opacity=.85 />`});s+=`<line x1=10 x2=382 y1=${H/2} y2=${H/2} stroke="#3f3f46"/>`}
else if(kind=='mom'){for(let i=0;i<14;i++)s+=`<circle cx=${30+i*25} cy=${H-20-(i*6+r()*30)} r=4 fill="${i>9?T:'#52525b'}"/>`}
else{const p=path(walk(90,r,2),W,H,14);s+=`<path d="${p}" stroke="${T}" fill=none stroke-width=1.6 />`}
return s}
window.QNPreview={preview,rnd,walk,path};
})();
