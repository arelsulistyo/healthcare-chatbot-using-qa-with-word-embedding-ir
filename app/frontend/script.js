const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendButton = document.getElementById("sendButton");

// Add event listeners when the document loads
document.addEventListener("DOMContentLoaded", () => {
  // Prevent any possible form submission
  document.addEventListener(
    "submit",
    (e) => {
      e.preventDefault();
      return false;
    },
    true
  );

  // Update send button click handler
  sendButton.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    sendMessage();
    return false;
  };

  // Update Enter key handler
  userInput.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();
      sendMessage();
      return false;
    }
  };

  // Load initial chat history
  createNewChat();
  loadChatHistory();
});

// Add these variables at the top
let currentChatId = null;
const generateChatId = () => Date.now().toString();

async function loadChatHistory(shouldLoadMessages = true) {
  try {
    const response = await fetch("http://localhost:8000/chat-history");
    const data = await response.json();

    if (data.status === "success") {
      // Update sidebar without affecting current messages
      updateHistorySidebar(data.history);
    }
  } catch (error) {
    console.error("Error loading chat history:", error);
  }
}

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

function addMessage(text, isUser) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user-message" : "bot-message"}`;

  if (isUser) {
    messageDiv.textContent = text;
  } else {
    // Split the response into chunks for smoother animation
    const formattedText = formatMessage(text);
    messageDiv.innerHTML = formattedText;
  }

  // Add the message with a slight delay for smoother appearance
  setTimeout(
    () => {
      chatMessages.appendChild(messageDiv);

      // Smooth scroll to the new message
      messageDiv.scrollIntoView({ behavior: "smooth", block: "end" });
    },
    isUser ? 0 : 100
  ); // Slight delay for bot messages
}

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

// Update sendMessage function - remove the event parameter
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return false;

  if (!currentChatId) {
    currentChatId = generateChatId();
  }

  // Store the message and clear input immediately
  const messageToSend = message;
  userInput.value = "";
  setInputState(true);

  // Add user message to chat
  addMessage(messageToSend, true);
  const loadingDiv = showLoading();

  try {
    const requestBody = {
      query: messageToSend,
      chatId: currentChatId,
    };

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(requestBody),
      redirect: "manual",
      mode: "cors",
      credentials: "same-origin",
    });

    // Check if we got redirected
    if (response.type === "opaqueredirect") {
      throw new Error("Redirect detected");
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      throw new Error("Received non-JSON response from server");
    }

    const data = await response.json();

    // Remove loading animation
    if (loadingDiv) {
      loadingDiv.remove();
    }

    if (data.status === "success") {
      addMessage(data.response, false);
      await loadChatHistory(false);
    } else {
      addMessage(`Error: ${data.response}`, false);
    }
  } catch (error) {
    console.error("Error in sendMessage:", error);
    if (loadingDiv) {
      loadingDiv.remove();
    }
    addMessage("Error connecting to the server. Please try again.", false);
  } finally {
    setInputState(false);
  }

  return false; // Ensure we return false
}

function formatTimestamp(isoString) {
  const date = new Date(isoString);
  return date.toLocaleString();
}

function updateHistorySidebar(history) {
  const historyList = document.getElementById("historyList");
  historyList.innerHTML = "";

  // Group messages by conversation
  const conversations = new Map();

  history.forEach((message) => {
    const chatId = message.chatId || "default";
    if (!conversations.has(chatId)) {
      conversations.set(chatId, []);
    }
    conversations.get(chatId).push(message);
  });

  // Create history items for each conversation
  Array.from(conversations.entries())
    .reverse()
    .forEach(([chatId, messages]) => {
      const firstMessage = messages.find((m) => m.role === "user");
      if (!firstMessage) return;

      const historyItem = document.createElement("div");
      historyItem.className = "history-item";
      if (chatId === currentChatId) {
        historyItem.classList.add("active");
      }

      historyItem.innerHTML = `
      <i class="fas fa-comments"></i>
      <div class="content">
        <p>${firstMessage.content}</p>
        <div class="timestamp">${formatTimestamp(firstMessage.timestamp)}</div>
      </div>
    `;

      historyItem.onclick = () => {
        currentChatId = chatId;
        chatMessages.innerHTML = "";
        // Only load messages for the selected chat
        messages.forEach((message) => {
          addMessage(message.content, message.role === "user");
        });

        document.querySelectorAll(".history-item").forEach((item) => {
          item.classList.remove("active");
        });
        historyItem.classList.add("active");
      };

      historyList.appendChild(historyItem);
    });
}

// Add clear history function
async function clearHistory() {
  if (!confirm("Are you sure you want to clear all chat history?")) return;

  try {
    const response = await fetch("http://localhost:8000/clear-history", {
      method: "POST",
    });
    const data = await response.json();

    if (data.status === "success") {
      chatMessages.innerHTML = "";
    }
  } catch (error) {
    console.error("Error clearing chat history:", error);
  }
}

// Add new chat function
function createNewChat() {
  currentChatId = generateChatId();
  chatMessages.innerHTML = "";

  // Update active state in sidebar
  document.querySelectorAll(".history-item").forEach((item) => {
    item.classList.remove("active");
  });

  // Add new chat to history if it's not already there
  const historyList = document.getElementById("historyList");
  const newChatItem = document.createElement("div");
  newChatItem.className = "history-item active";
  newChatItem.innerHTML = `
    <i class="fas fa-comments"></i>
    <div class="content">
      <p>New Chat</p>
      <div class="timestamp">${new Date().toLocaleString()}</div>
    </div>
  `;
  historyList.insertBefore(newChatItem, historyList.firstChild);
}
