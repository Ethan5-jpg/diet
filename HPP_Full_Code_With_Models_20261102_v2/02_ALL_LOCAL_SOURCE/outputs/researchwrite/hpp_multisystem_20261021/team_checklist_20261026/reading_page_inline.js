
const cards=[...document.querySelectorAll('details.task')];
document.getElementById('expand').addEventListener('click',()=>cards.forEach(x=>x.open=true));
document.getElementById('collapse').addEventListener('click',()=>cards.forEach(x=>x.open=false));
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.trim().toLocaleLowerCase();cards.forEach(x=>{const match=!q||x.textContent.toLocaleLowerCase().includes(q);x.hidden=!match;if(q&&match)x.open=true;});});
window.addEventListener('hashchange',()=>{const x=document.getElementById(location.hash.slice(1));if(x&&x.matches('details'))x.open=true;});
document.getElementById('print').addEventListener('click',()=>{cards.forEach(x=>{x.hidden=false;x.open=true;});window.print();});
