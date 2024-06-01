
<!-- giteement评论系统 -->
//document.write("<script language=javascript src="+url+"></script>");
document.write("<link rel=\"stylesheet\" href=\"https://giteement.oss-cn-beijing.aliyuncs.com/default.css\">");
let div = document.createElement("div");
div.setAttribute("id","giteementDiv");
let body = document.getElementsByTagName("body")[0];
body.append(div);
var giteement = new Giteement({
  id: 'forum',
  owner: 'putik-666',
  repo: 'blogs',
  backcall_uri: 'https://putik-666.gitee.io',
  oauth_uri: 'https://cors-anywhere.herokuapp.com/https://gitee.com/oauth/token',
  oauth: {
    client_id: '20c39b412c4e4c6832fd6e9884040c03856665d3e7d84a29437217b0ca554d6f',
    client_secret: '81c7619ccbe096266c7dc6f3c0494a8564fd1ee5baf152853bb78265193e1a56'
  },
});
giteement.render('giteementDiv');