function uploadVideo() {
    const fileInput = document.getElementById('videoUpload');
    const videoSection = document.getElementById('videoSection');
    const videoStream = document.getElementById('videoStream');

    if (!fileInput.files.length) {
        alert('Please select a video file.');
        return;
    }

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Show the video section
        videoSection.classList.remove('hidden');
        videoStream.src = `/stream/${encodeURIComponent(data.filepath)}`;

        // Listen for cardiac arrest detection messages via SSE
        const eventSource = new EventSource('/status');
        let alertShown = false;

        eventSource.onmessage = function(event) {
            if (event.data === 'Cardiac Arrest Detected' && !alertShown) {
                showAlert();
                alertShown = true;
                eventSource.close(); // Close the connection after showing the alert
            }
        };

        eventSource.onerror = function() {
            console.error('EventSource failed.');
            eventSource.close();
        };
    })
    .catch(error => {
        console.error('Error uploading video:', error);
        alert('Failed to upload video.');
    });
}

function showAlert() {
    const alertModal = document.getElementById('alertModal');
    alertModal.classList.remove('hidden');
}

function closeAlert() {
    const alertModal = document.getElementById('alertModal');
    alertModal.classList.add('hidden');
}