var threshold = 50;//if sync.get fails, we use 50 as a default.

chrome.storage.sync.get({
  threshold: '50'
}, function(items) {
  threshold = items.threshold;
});

//deal with newly loaded tweets
function DOMModificationHandler(){
    $(this).unbind('DOMSubtreeModified.event1');
    setTimeout(function(){
        modify();
        $('#timeline').bind('DOMSubtreeModified.event1',DOMModificationHandler);
    },10);
}
$('#timeline').bind('DOMSubtreeModified.event1',DOMModificationHandler);


function modify(){
  //find and modify tall tweets
  $('.tweet-text').each(function(index){
    var type = 99
    var tweet = $(this)
    var t = tweet.html();
    var tweetText = tweet[0].innerText;
    if(tweetText.indexOf("This tweet has been marked as") == 0 ) {
      return
    }
    $.ajax({
      url: "https://virtus-server.herokuapp.com/predict",
      // url: "http://localhost:4000/api",
      type: "POST",
      data: {
          tweet: tweetText
      },
      success: function(response){
          switch (response.category) {
            case 0: type = "Racist"; break;
            case 1: type = "Sexist"; break;
            case 2: type = "Hate"; break;
            case 3: type = "Offensive"; break;
            case 4: type = "Neutral"; break;
          }
          if(!tweet.hasClass("squished") && response.category > 0 && response.category < 4){
            tweet.addClass("squished");
            tweet.html(`<p>This tweet has been marked as <b>${type}</b> </p> <button class="squish-button EdgeButton EdgeButton--primary" data-original-content="${encodeURI(t)}">Show Offensive Tweet</button>`);
            //if we add a new button, we have to add listeners again...
            chrome.runtime.sendMessage({message: "listeners"}, function(response) {
            });
          }
      }
    });
  });
}
