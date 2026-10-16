// GA4: eventos do Quant Notebooks (page_view vem do gtag config; scroll via Enhanced Measurement)
(function(){
var g=function(n,p){if(window.gtag)gtag('event',n,p||{})};
var nb=(window.QN_NB&&window.QN_NB.file||'').replace(/^notebooks\//,'').replace(/\.py$/,'')||'home';
var txt=function(el){return (el.textContent||'').trim().slice(0,100)};
document.addEventListener('click',function(e){
 var t=e.target.closest&&e.target.closest('a,button');if(!t)return;
 if(t.closest('.qn-toc')&&t.tagName=='A'){g('toc_click',{section:txt(t),notebook:nb});return}
 if(t.classList.contains('qn-code-toggle')){var c=t.closest('.marimo-cell');if(c&&!c.classList.contains('qn-collapsed')){var m=/célula (\d+)/.exec(t.textContent);g('code_expand',{cell:m?+m[1]:0,notebook:nb})}return}
 if(t.dataset.themeF||t.dataset.levelF){g('gallery_filter',{filter_type:t.dataset.themeF?'theme':'level',filter_value:t.dataset.themeF||t.dataset.levelF});return}
 if(t.matches('a.card,a.feat')){var p=(t.getAttribute('href')||'').replace(/^notebooks\//,'').replace(/\.html$/,'');g('notebook_open',{notebook:p,source:t.classList.contains('feat')?'featured':'gallery'});return}
 if(t.tagName=='A'&&t.href){var u;try{u=new URL(t.href)}catch(_){return}
  var dl=t.hasAttribute('download');
  if(dl||u.host!==location.host||!u.pathname.startsWith('/quant-notebooks/')){
   var type=dl?'download':/github\.com$/.test(u.host)?'github':u.pathname.startsWith('/monitor-b3')?'monitor_b3':u.host===location.host?'portfolio':'external';
   g('outbound_click',{link_url:t.href,link_type:type,link_text:txt(t),notebook:nb,transport_type:'beacon'})}}
},false);
document.addEventListener('copy',function(e){var s=window.getSelection&&window.getSelection();var n=s&&s.anchorNode;var el=n&&(n.nodeType==1?n:n.parentElement);if(el&&el.closest('.qn-cmd'))g('copy_run_command',{notebook:nb})});
var q,qt;document.addEventListener('input',function(e){if(e.target.id!=='q')return;clearTimeout(qt);qt=setTimeout(function(){var v=e.target.value.trim();if(v&&v!==q){q=v;g('gallery_filter',{filter_type:'search',filter_value:v.slice(0,50)})}},1200)});
})();
