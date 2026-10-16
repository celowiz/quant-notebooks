(function(){
const M=window.QN_NB||{};const A=M.assets||'../../assets/';
document.body.classList.add('qn-nb');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
// barra superior
const nav=document.createElement('header');nav.className='qn-nav';
nav.innerHTML=`<a class="qn-logo" href="${A}../index.html"><span class="mw">m<b>w</b></span><span class="hide-sm">Marcelo Wizenberg</span><span class="sep hide-sm">/</span><span class="hide-sm">Quant Notebooks</span><span class="sep hide-sm">/</span><span class="cur">${esc(M.short||M.title||'')}</span></a><nav class="qn-links"><a class="hide-sm" href="${M.portfolio}">Portfólio</a><a class="on" href="${A}../index.html">Notebooks</a><a href="${M.repo}" target="_blank" rel="noopener">GitHub ↗</a></nav>`;
document.body.prepend(nav);
const ghBtn=`<a class="btn p" href="${M.source}" target="_blank" rel="noopener">Ver código no GitHub ↗</a>`;
const dlBtn=`<a class="btn" href="${M.raw}" download>Baixar .py</a>`;
const backBtn=`<a class="btn" href="${A}../index.html">← Todos os notebooks</a>`;
// trilho direito
const side=document.createElement('aside');side.className='qn-rail qn-side';
side.innerHTML=`<section><div class="eyebrow">Executar</div>${ghBtn}${dlBtn}<p class="disclaimer" style="margin:6px 0 10px">Esta é a versão estática: gráficos e tabelas estão prontos, mas controles interativos só funcionam rodando o notebook.</p><div class="ill" style="margin:10px 0 2px">Rodar localmente</div><code class="qn-cmd">git clone ${esc(M.repo)}.git
cd quant-notebooks
uv run marimo edit ${esc(M.file)}</code></section>
<section><div class="eyebrow">Tags</div>${(M.tags||[]).map(t=>`<span class="chip static">${esc(t)}</span>`).join('')}</section>
<section><div class="eyebrow">Relacionados</div>${(M.related||[]).map(r=>`<a class="rel" href="${A}../${r.path}.html">${esc(r.title)} →</a>`).join('')}</section>
<section><p class="disclaimer">Conteúdo educacional. Nada aqui é recomendação de investimento.</p></section>`;
document.body.append(side);
const toc=document.createElement('aside');toc.className='qn-rail qn-toc';toc.innerHTML=`<div class="eyebrow">Neste notebook</div><ol></ol><div class="ill">progresso de leitura</div><div class="qn-prog"><i></i></div>`;document.body.append(toc);
const mb=document.createElement('div');mb.className='qn-mobile-bar';mb.innerHTML=backBtn+ghBtn.replace('btn p','btn')+dlBtn;document.body.append(mb);
let tries=0;
// esconde o toast "Static notebook" do marimo (o aviso equivalente fica no trilho lateral)
new MutationObserver(()=>document.querySelectorAll('li[data-radix-collection-item]').forEach(li=>{if(/^Static notebook/.test(li.textContent.trim()))li.classList.add('qn-hide')})).observe(document.body,{childList:true,subtree:true});
function enhance(){const app=document.getElementById('App');const cells=app?app.querySelectorAll('.marimo-cell'):[];
 const eds=app?[...app.querySelectorAll('.cm-content')]:[];
 if((!cells.length||eds.some(e=>!e.textContent.trim()))&&tries++<40)return setTimeout(enhance,250);if(!cells.length)return;
 // banner "Static marimo notebook"
 const cand=[...document.querySelectorAll('#root div')].filter(el=>{const t=(el.textContent||'').trim();return t.length<130&&/^Static\s*marimo notebook/i.test(t)});
 cand.filter(el=>!cand.includes(el.parentElement)).forEach(el=>el.classList.add('qn-hide'));
 let n=0;cells.forEach(cell=>{const ed=cell.querySelector('.cm-content');if(!ed)return;const ls=[...ed.querySelectorAll('.cm-line')].map(l=>l.textContent);const code=(ls.length?ls.join('\n'):ed.textContent).trim();
  const out=cell.querySelector('[id^="output-"]');const hasOut=out&&out.innerText.trim().length>0||out&&out.querySelector('img,svg,canvas,iframe,table');
  if(/^import marimo as mo$/.test(code)){cell.classList.add('qn-hide');return}
  n++;const lines=code.split('\n').length;const tray=cell.querySelector('.tray');if(!tray)return;
  const b=document.createElement('button');b.className='qn-code-toggle';b.type='button';
  const lab=()=>`<span>${cell.classList.contains('qn-collapsed')?'▸ mostrar código':'▾ ocultar código'} · célula ${n}</span><span>python · ${lines} linha${lines>1?'s':''}</span>`;
  cell.classList.add('qn-collapsed');b.innerHTML=lab();b.onclick=()=>{cell.classList.toggle('qn-collapsed');b.innerHTML=lab()};tray.parentElement.insertBefore(b,tray)});
 // cabeçalho editorial antes do primeiro h1
 const h1=app.querySelector('.markdown h1');
 if(h1&&!document.querySelector('.qn-head')){const h=document.createElement('div');h.className='qn-head';h.innerHTML=`<span class="eyebrow">${esc(M.theme||'')} · ${esc(M.level||'')}</span>Marcelo Wizenberg · ${esc(M.date||'')} · ◷ ${esc(M.minutes||'')} min de leitura · Python · marimo`;h1.parentElement.insertBefore(h,h1);
  if(M.title&&M.title!==h1.textContent.trim()&&M.subtitle){}}
 // TOC
 const ol=toc.querySelector('ol');const hs=[...app.querySelectorAll('.markdown h1,.markdown h2')];
 hs.forEach((h,i)=>{if(!h.id)h.id='sec-'+i;const li=document.createElement('li');li.className=h.tagName=='H1'?'l1':'l2';li.innerHTML=`<a href="#${h.id}">${esc(h.tagName=='H1'?'Introdução':h.textContent.trim())}</a>`;li.firstChild.onclick=e=>{e.preventDefault();h.scrollIntoView({behavior:'smooth',block:'start'})};ol.append(li)});
 const bar=toc.querySelector('.qn-prog i');const links=[...ol.querySelectorAll('a')];
 function onScroll(){const max=app.scrollHeight-app.clientHeight;bar.style.width=(max>0?app.scrollTop/max*100:100)+'%';let cur=0;hs.forEach((h,i)=>{if(h.getBoundingClientRect().top<160)cur=i});links.forEach((a,i)=>a.classList.toggle('on',i==cur))}
 app.addEventListener('scroll',onScroll,{passive:true});onScroll()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(enhance,300));else setTimeout(enhance,300);
})();
