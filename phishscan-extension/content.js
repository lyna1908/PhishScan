chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    if (request.action === "extractEmail") {

        try {

            // Subject
            let subjectElement = document.querySelector("h2");

            // Sender
            let senderElement = document.querySelector(".gD");

            // Body
            let bodyElement = document.querySelector(".a3s.aiL");

            // Extract values
            let subject = subjectElement
                ? subjectElement.innerText
                : "";

            let senderEmail = senderElement
                ? senderElement.getAttribute("email")
                : "";

            let body = bodyElement
                ? bodyElement.innerText
                : "";

            console.log("=== EMAIL EXTRACTED ===");

            console.log({
                sender: senderEmail,
                subject: subject,
                body: body
            });

            // IMPORTANT
            sendResponse({
                sender: senderEmail,
                subject: subject,
                body: body
            });

        } catch (error) {

            console.error("Extraction Error:", error);

            sendResponse(null);
        }
    }

    return true;
});