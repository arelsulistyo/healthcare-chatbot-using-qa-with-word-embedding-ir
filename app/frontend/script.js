const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendButton = document.getElementById("sendButton");

// Add event listener for Enter key
userInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});

function formatMessage(text) {
  // Handle bold text with ** or *
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*(.*?)\*/g, "<strong>$1</strong>");

  // Handle bullet points
  text = text
    .split("\n")
    .map((line) => {
      if (line.trim().startsWith("*")) {
        return `<li>${line.substring(1).trim()}</li>`;
      }
      return `<p>${line}</p>`;
    })
    .join("");

  // Wrap bullet points in ul
  text = text.replace(/<li>.*?<\/li>/gs, (match) => `<ul>${match}</ul>`);

  return text;
}

function addMessage(text, isUser, context = null) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user-message" : "bot-message"}`;

  if (isUser) {
    messageDiv.textContent = text;
  } else {
    const formattedText = formatMessage(text);
    messageDiv.innerHTML = formattedText;

    // Add context button if context exists
    if (context) {
      const contextButton = document.createElement("button");
      contextButton.className = "context-button";
      contextButton.textContent = "Show Context";
      contextButton.onclick = () => showContext(context);
      messageDiv.appendChild(contextButton);
    }
  }

  setTimeout(
    () => {
      chatMessages.appendChild(messageDiv);
      messageDiv.scrollIntoView({ behavior: "smooth", block: "end" });
    },
    isUser ? 0 : 100
  );
}

function showContext(context) {
  const modal = document.getElementById("contextModal");
  const qaPairsDiv = document.getElementById("qaPairs");
  const abstractsDiv = document.getElementById("abstracts");

  // Clear previous content
  qaPairsDiv.innerHTML = "";
  abstractsDiv.innerHTML = "";

  // Add QA pairs
  context.qa_pairs.forEach((qa) => {
    const qaDiv = document.createElement("div");
    qaDiv.className = "qa-pair";
    qaDiv.innerHTML = `
      <h4>Q: ${qa.Question}</h4>
      <p>A: ${qa.Answer}</p>
    `;
    qaPairsDiv.appendChild(qaDiv);
  });

  // Add abstracts
  context.abstracts.forEach((abstract) => {
    const abstractDiv = document.createElement("div");
    abstractDiv.className = "abstract";
    abstractDiv.innerHTML = `<p>${abstract.abstract_text}</p>`;
    abstractsDiv.appendChild(abstractDiv);
  });

  modal.style.display = "block";
}

// Add modal close functionality
document.querySelector(".close").onclick = function () {
  document.getElementById("contextModal").style.display = "none";
};

window.onclick = function (event) {
  const modal = document.getElementById("contextModal");
  if (event.target == modal) {
    modal.style.display = "none";
  }
};

function showLoading() {
  const loadingDiv = document.createElement("div");
  loadingDiv.className = "loading";
  loadingDiv.innerHTML = `
    <div class="loading-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;

  // Insert loading dots at the bottom of the chat
  chatMessages.appendChild(loadingDiv);
  loadingDiv.scrollIntoView({ behavior: "smooth", block: "end" });
  return loadingDiv;
}

// Add this new function to handle input state
function setInputState(disabled) {
  userInput.disabled = disabled;
  sendButton.disabled = disabled;
  if (!disabled) {
    userInput.focus();
  }
}

// Update the sendMessage function
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  userInput.value = "";
  setInputState(true);

  addMessage(message, true);
  const loadingDiv = showLoading();

  try {
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: message }),
    });

    const data = await response.json();

    // Smooth removal of loading indicator
    loadingDiv.style.opacity = "0";
    setTimeout(() => loadingDiv.remove(), 100);

    if (data.status === "success") {
      const context = {
        qa_pairs: data.qa_pairs,
        abstracts: data.abstracts,
      };
      addMessage(data.response, false, context);
    } else {
      addMessage("Sorry, there was an error processing your request.", false);
    }
  } catch (error) {
    loadingDiv.style.opacity = "0";
    setTimeout(() => loadingDiv.remove(), 300);
    addMessage("Error connecting to the server. Please try again.", false);
  } finally {
    setInputState(false);
  }
}
