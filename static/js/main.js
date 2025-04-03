document.getElementById('detailedAnalysisBtn').addEventListener('click', function() {
    const audioFilePath = "{{ audio_file | tojson | safe }}";
    if (!audioFilePath) {
        console.error('Audio file path is not available.');
        return;
    }
    fetch('/detailed_analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file: audioFilePath })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        document.getElementById('analysisReport').style.display = 'block';
        const ctx = document.getElementById('emotionGraph').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.time_intervals,
                datasets: [{
                    label: 'Emotion Intensity',
                    data: data.emotion_intensity,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Emotion Intensity'
                        }
                    }
                }
            }
        });

        const tableBody = document.getElementById('emotionChangeTable').querySelector('tbody');
        tableBody.innerHTML = '';
        data.time_intervals.forEach((interval, index) => {
            const row = document.createElement('tr');
            const timeCell = document.createElement('td');
            const emotionCell = document.createElement('td');
            timeCell.textContent = interval;
            emotionCell.textContent = data.emotion_intensity[index];
            row.appendChild(timeCell);
            row.appendChild(emotionCell);
            tableBody.appendChild(row);
        });

        document.getElementById('downloadReportBtn').href = data.report_url;
    })
    .catch(error => console.error('Error:', error));
});
