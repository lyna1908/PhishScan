document.getElementById("scanBtn").addEventListener("click", async () => {

    const resultDiv = document.getElementById("result");

    const detailsDiv = document.getElementById("details");
    const toggleBtn = document.getElementById("toggleDetailsBtn");

    const loadingDiv = document.getElementById("loading");

    resultDiv.innerHTML = "";
    detailsDiv.innerHTML = "";

    loadingDiv.classList.remove("hidden");

    const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    chrome.tabs.sendMessage(
        tab.id,
        { action: "extractEmail" },

        async (emailData) => {

            loadingDiv.classList.add("hidden");

            if (!emailData) {

                resultDiv.className = "phishing";

                resultDiv.innerHTML =
                    "Could not extract email.";

                return;
            }

            try {

                const response = await fetch(
                    "http://localhost:5000/analyze",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify(emailData)
                    }
                );

                const data = await response.json();

                if (data.prediction === "phishing") {

                    resultDiv.className = "phishing";

                    resultDiv.innerHTML =
                        `⚠️ PHISHING (${data.probability}%)`;

                } else {

                    resultDiv.className = "safe";

                    resultDiv.innerHTML =
                        `✅ SAFE (${data.probability}%)`;
                }

                let detailsHTML =
    "<h3>Detailed Analysis</h3>";

if (data.details.length === 0) {

    detailsHTML +=
        `<div class="detail-item">
            No suspicious indicators detected.
        </div>`;

} else {

    data.details.forEach(item => {

        detailsHTML +=
            `<div class="detail-item">
                • ${item}
            </div>`;
    });
}

                detailsDiv.innerHTML = detailsHTML;
		toggleBtn.classList.remove("hidden");
detailsDiv.classList.add("hidden");

toggleBtn.innerText = "Show Details";
            } catch (error) {

                resultDiv.className = "phishing";

                resultDiv.innerHTML =
                    "Connection to API failed.";

                console.error(error);
            }
        }
    );
});
document
    .getElementById("toggleDetailsBtn")
    .addEventListener("click", () => {

        const details =
            document.getElementById("details");

        const btn =
            document.getElementById("toggleDetailsBtn");

        if (details.classList.contains("hidden")) {

            details.classList.remove("hidden");

            btn.innerText = "Hide Details";

        } else {

            details.classList.add("hidden");

            btn.innerText = "Show Details";
        }
    });