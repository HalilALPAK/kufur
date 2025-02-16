chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'checkSwearWords') {
        // Video ID'si alındı
        let videoId = request.videoId;

        // Sunucuya video ID'sini gönder
        fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ video_id: videoId })
        })
        .then(response => response.json())
        .then(data => sendResponse(data))
        .catch(error => console.error('Error:', error));

        return true;  // Asenkron işleme devam etmesi için true döndürülmeli
    }
});
