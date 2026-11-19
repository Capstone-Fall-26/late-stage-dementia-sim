(function(){
var gd=document.getElementById('memory-map'),stage=document.getElementById('stage');
function lay(t,id){var n=t.name||'',g=t.legendgroup||'';
 if(id==='episodic')return n.indexOf('episodic memory')>=0;
 if(id==='facts')return n.indexOf('biographical fact')>=0;
 if(id==='clusters')return n==='Cluster spokes'||n==='Cluster hub';
 if(id==='drift')return g==='drift';
 if(id==='query')return g==='query';return false;}
document.querySelectorAll('.chip').forEach(function(c){c.addEventListener('click',function(){
 var on=c.getAttribute('aria-pressed')!=='true',id=c.dataset.layer;c.setAttribute('aria-pressed',on);
 var idx=[];gd.data.forEach(function(t,i){if(lay(t,id))idx.push(i);});
 if(idx.length)Plotly.restyle(gd,{visible:on},idx);
 if(id==='clusters'){var u={};(gd.layout.annotations||[]).forEach(function(a,i){u['annotations['+i+'].visible']=on;});Plotly.relayout(gd,u);}
});});
function zoom(f){var x=gd._fullLayout.xaxis.range,y=gd._fullLayout.yaxis.range,
 cx=(x[0]+x[1])/2,cy=(y[0]+y[1])/2,hx=(x[1]-x[0])/2*f,hy=(y[1]-y[0])/2*f;
 Plotly.relayout(gd,{'xaxis.range':[cx-hx,cx+hx],'yaxis.range':[cy-hy,cy+hy]});}
var act={'zoom-in':function(){zoom(.7);},'zoom-out':function(){zoom(1.4);},
 'reset':function(){Plotly.relayout(gd,{'xaxis.autorange':true,'yaxis.autorange':true});},
 'png':function(){Plotly.downloadImage(gd,{format:'png',filename:'memory_map',width:1600,height:900,scale:2});},
 'full':function(){if(document.fullscreenElement)document.exitFullscreen();else stage.requestFullscreen();},
 'pan':function(b){var on=b.getAttribute('aria-pressed')!=='true';b.setAttribute('aria-pressed',on);
  Plotly.relayout(gd,{dragmode:on?'pan':'zoom'});}};
document.querySelectorAll('.tool').forEach(function(b){b.addEventListener('click',function(){act[b.dataset.act](b);});});
document.addEventListener('fullscreenchange',function(){
 Plotly.relayout(gd,{height:document.fullscreenElement?window.innerHeight-70:700});});
})();
