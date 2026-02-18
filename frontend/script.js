async function uploadFile() {

    const role = document.getElementById("roleSelect").value;
    const file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Please upload an image or video file.");
        return;
    }

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    let endpoint = file.type.startsWith("video")
        ? "http://127.0.0.1:8000/detect-video-with-heatmap/"
        : "http://127.0.0.1:8000/detect-image/";

    try {

        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("results").classList.remove("hidden");

        // Reset sections
        document.querySelector(".analystOnly").classList.add("hidden");
        document.querySelector(".researcherOnly").classList.add("hidden");

        // Image Results
        if (data.original_base64) {
            document.getElementById("originalImg").src =
                "data:image/jpeg;base64," + data.original_base64;

            document.getElementById("enhancedImg").src =
                "data:image/jpeg;base64," + data.enhanced_base64;

            document.getElementById("detectedImg").src =
                "data:image/jpeg;base64," + data.detected_base64;
        }

        // Heatmap (Video)
        if (data.heatmap_image_base64) {
            document.getElementById("heatmapImg").src =
                "data:image/png;base64," + data.heatmap_image_base64;
        }

        // Role Display Logic
        if (role === "analyst" || role === "researcher") {
            document.querySelector(".analystOnly").classList.remove("hidden");
        }

        if (role === "researcher" && data.total_objects !== undefined) {

            let statsHTML = `
                <p><strong>Total Objects:</strong> ${data.total_objects}</p>
                <p><strong>Average Confidence:</strong> ${data.average_confidence}</p>
            `;

            for (let key in data.class_counts) {
                statsHTML += `
                    <p>${key}: ${data.class_counts[key]} 
                    (${data.class_percentages[key]}%)</p>
                `;
            }

            document.getElementById("stats").innerHTML = statsHTML;
            document.querySelector(".researcherOnly").classList.remove("hidden");
        }

    } catch (error) {
        alert("Error processing file.");
        console.error(error);
    }
}
