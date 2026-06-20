const loginForm = document.querySelector("#login-form");
const bootstrapForm = document.querySelector("#bootstrap-form");
const account = document.querySelector("#account");
const adminPanel = document.querySelector("#admin-panel");
const message = document.querySelector("#message");

const roleLabels = {
  admin: "Administrateur",
  analyste: "Analyste",
  conseiller: "Conseiller",
};

function showMessage(text, success = false) {
  message.textContent = text;
  message.classList.toggle("success", success);
  message.hidden = false;
}

function clearMessage() {
  message.hidden = true;
  message.textContent = "";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = "Une erreur est survenue.";
    try {
      const data = await response.json();
      detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join(" ")
        : data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function showAccount(user) {
  loginForm.hidden = true;
  bootstrapForm.hidden = true;
  account.hidden = false;
  document.querySelector("#welcome").textContent = `Bienvenue, ${user.username}`;
  document.querySelector("#account-email").textContent = user.email;
  document.querySelector("#role-badge").textContent = roleLabels[user.role] || user.role;
  adminPanel.hidden = user.role !== "admin";
  if (user.role === "admin") loadUsers();
}

function showLogin() {
  account.hidden = true;
  adminPanel.hidden = true;
  bootstrapForm.hidden = true;
  loginForm.hidden = false;
}

async function initialize() {
  clearMessage();
  try {
    const user = await api("/auth/me");
    showAccount(user);
    return;
  } catch (_) {}

  try {
    const status = await api("/auth/status");
    loginForm.hidden = !status.initialized;
    bootstrapForm.hidden = status.initialized;
  } catch (error) {
    showMessage(error.message);
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();
  try {
    const data = await api("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: document.querySelector("#login-username").value,
        password: document.querySelector("#login-password").value,
      }),
    });
    showAccount(data.user);
  } catch (error) {
    showMessage(error.message);
  }
});

bootstrapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();
  try {
    await api("/auth/bootstrap", {
      method: "POST",
      body: JSON.stringify({
        username: document.querySelector("#bootstrap-username").value,
        email: document.querySelector("#bootstrap-email").value,
        password: document.querySelector("#bootstrap-password").value,
        role: "admin",
      }),
    });
    showMessage("Administrateur créé. Vous pouvez maintenant vous connecter.", true);
    showLogin();
  } catch (error) {
    showMessage(error.message);
  }
});

document.querySelector("#logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
  showLogin();
  showMessage("Vous êtes déconnecté.", true);
});

document.querySelectorAll(".reveal").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.querySelector(`#${button.dataset.target}`);
    input.type = input.type === "password" ? "text" : "password";
    button.textContent = input.type === "password" ? "Afficher" : "Masquer";
  });
});

async function loadUsers() {
  try {
    const users = await api("/admin/users");
    const body = document.querySelector("#users-body");
    body.innerHTML = "";
    users.forEach((user) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${escapeHtml(user.username)}</strong><br><span class="muted">${escapeHtml(user.email)}</span></td>
        <td>
          <select class="role-select" data-user-id="${user.id}">
            ${Object.entries(roleLabels).map(([value, label]) => `<option value="${value}" ${user.role === value ? "selected" : ""}>${label}</option>`).join("")}
          </select>
        </td>
        <td class="${user.is_active ? "status-active" : "status-inactive"}">${user.is_active ? "Actif" : "Désactivé"}</td>
        <td><button class="small-button active-toggle" data-user-id="${user.id}" data-active="${user.is_active}">${user.is_active ? "Désactiver" : "Activer"}</button></td>
      `;
      body.appendChild(row);
    });
    bindUserActions();
  } catch (error) {
    showMessage(error.message);
  }
}

function bindUserActions() {
  document.querySelectorAll(".role-select").forEach((select) => {
    select.addEventListener("change", async () => {
      try {
        await api(`/admin/users/${select.dataset.userId}/role`, {
          method: "PATCH",
          body: JSON.stringify({ role: select.value }),
        });
        showMessage("Rôle mis à jour.", true);
      } catch (error) {
        showMessage(error.message);
        loadUsers();
      }
    });
  });
  document.querySelectorAll(".active-toggle").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/admin/users/${button.dataset.userId}/active`, {
          method: "PATCH",
          body: JSON.stringify({ is_active: button.dataset.active !== "true" }),
        });
        await loadUsers();
      } catch (error) {
        showMessage(error.message);
      }
    });
  });
}

document.querySelector("#create-user-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/admin/users", {
      method: "POST",
      body: JSON.stringify({
        username: document.querySelector("#new-username").value,
        email: document.querySelector("#new-email").value,
        password: document.querySelector("#new-password").value,
        role: document.querySelector("#new-role").value,
      }),
    });
    event.target.reset();
    showMessage("Utilisateur créé.", true);
    await loadUsers();
  } catch (error) {
    showMessage(error.message);
  }
});

document.querySelector("#refresh-users").addEventListener("click", loadUsers);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

initialize();
