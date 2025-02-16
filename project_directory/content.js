// YouTube sayfasında oynatılan video ID'sini al
let videoId = window.location.href.split("v=")[1].split("&")[0];

// videoId'yi arka plana (background.js) göndermek için message gönder
chrome.runtime.sendMessage({ action: 'checkSwearWords', videoId: videoId });
