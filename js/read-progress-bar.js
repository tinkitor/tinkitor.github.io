//先判断是否为文章页
let NeedProgressBar = window.location.pathname.includes("article");
if(NeedProgressBar){
	document.addEventListener('DOMContentLoaded', function () {
    		var winHeight = window.innerHeight,
        	docHeight = document.documentElement.scrollHeight,
        	progressBar = document.querySelector('#content_progress');
    		progressBar.max = docHeight - winHeight;
    		progressBar.value = window.scrollY;

    		document.addEventListener('scroll', function () {
       			progressBar.max = document.documentElement.scrollHeight - window.innerHeight;
        		progressBar.value = window.scrollY;
    		});
	});	
}
