# Team Server

The Team Server is the team edition of Perceptron: it runs on a server in your organization
(on-premise), is used from the browser with the same interface as the desktop, and adds users, roles,
shared projects and a training queue with CPU and GPU workers. It also serves as a training target for
desktops.

This page has two parts: [for users](#for-users) and [for administrators](#for-administrators).

## For users

### Signing in

Open in your browser the URL your Admin gave you (for example, `https://perceptron.company.com`).

- **With SSO:** click **Sign in with "provider"** (for example, Microsoft or Google). Two-factor
  authentication is handled by your identity provider.
- **With a server account:** fill in **Email** and **Password** and click **Sign in**.

After several failed attempts the account is locked for a few minutes. To leave, **Sign out** (top
right).

### Roles

| Role | What they can do |
|---|---|
| **Admin** | Everything an Editor can, plus users, roles, LLM and privacy policies, audit log and license |
| **Editor** | Create and edit projects, load data, train, evaluate, register, export and deploy |
| **Viewer** | View projects, runs, reports and models; use the playground. Cannot train, export or modify |

Roles are granted per **workspace** and per **project**, and the project role wins: you can be Editor in
one project and Viewer everywhere else. If you only have read access, **Read only** appears next to the
project name. Permissions are enforced by the server, not only by the screen.

### Shared projects

Workspace projects belong to the team: everyone with a role in it sees them in the sidebar, with their
data, runs and models. Every change is recorded in the server audit log.

### Data on the server

The web UI has no access to the folders on your machine. In **Data** you have two options:

- **Upload file** / **Upload folder** from the browser.
- **Server sources:** folders the Admin mounts on the server (read-only). Browse the **Path** and click
  **Use**. Handy for large datasets already on the network.

Databases and public datasets work just like on the desktop. See [Data](datos.md).

### Training queue

In the web UI, every training goes to the **queue** and a worker picks it up. The **Queue** page
(sidebar) shows:

- The mode and the **quotas**: how many running studies each person and each workspace may have.
- **Workers:** name, **Queues** (`gpu` / `cpu`), **Hardware** (GPU or "CPU only") and **State** (**Busy**
  / **Idle**). If no workers are connected, studies wait.
- **Queued or running studies:** study, queue, since when and on which worker it runs.

Live progress shows in **Experiments**, just like locally. Priorities between users are **coming soon**.

### Connecting the desktop to a server

This is how you train on the server GPU without leaving your desktop:

1. On the desktop, go to **Settings → Team servers**.
2. Fill in **Name** (how you will see it), **URL**, **Email** and **Password**, and click **Connect**. The
   password is only used to sign in: a token is kept in the keychain, never the password.
3. In a project, in **Train → step 4**, pick under **Where to train**: **On this machine** or **On
   "server name"**. Check **Use the server GPU** to go to a GPU worker.
4. Click **Train now**. Perceptron uploads the project, the data version, the pipeline and the
   architecture to the server ("Uploading to the server…"; the upload is chunked and resumes if
   interrupted) and brings the runs back.

To disconnect, use **Remove** in the server list. For now the desktop connection uses a server account
with a password (not SSO), and promoting a whole local project to a team project is **coming soon**.

## For administrators

Installation, SSO and backups are covered in the operations documentation:

- [Deployment (Docker Compose and Helm)](https://github.com/das2010/perceptron/blob/main/server/deploy/README.md)
- [SSO with Entra ID, Google Workspace or another OIDC provider](https://github.com/das2010/perceptron/blob/main/docs/ops/sso.md)
- [Backups and restore](https://github.com/das2010/perceptron/blob/main/docs/ops/backups.md)

### Administration console

**Administration** (sidebar) has four tabs. **Users** and **Audit log** are for the server Admin;
**Roles** and **Policies** are also available to workspace Admins.

#### Users

- **New user:** **Email**, **Name**, **Initial password** (at least 12 characters), **Workspace role**
  and, if applicable, **Server admin**. Click **Create user**.
- The list shows **Status** (Active, Inactive, Locked) and **Last sign-in**, and lets you **Activate** /
  **Deactivate**, **Make admin** / **Remove admin** or set a **New password**.
- With SSO, users are created on their first sign-in and their roles come from the identity provider
  groups.

#### Roles

Pick **Workspace**, **User**, **Scope** (**Whole workspace** or a project) and **Role**, and click
**Grant**. **Remove role** revokes it. A project role overrides the workspace role.

#### Policies

- **Allowed providers:** comma separated (for example, `ollama, anthropic`). Empty = all.
- **Maximum LLM privacy level:** no project in the workspace sends more than this level (L0 = nothing).
- **Maximum with a local LLM:** can be higher when the model runs on your own network (for example, L3
  with Ollama while the general level is L1).
- **SSO:** lists the configured providers. They are configured on the server, not from the console.

#### Audit log

Append-only log of sign-ins, writes, denied access, data reads, downloads and role changes. Filter by
**Action** (prefix, for example `auth.` or `api.createStudy`) and by **User**; you see **When**, **User**,
**Action**, **Resource** and **Result**. LLM calls are audited per project, under **LLM audit**.

Managing quotas, storage and worker status from the console is **coming soon**; today they are set in
the deployment.

### License

**Settings → License** shows the **Status** (Valid, No license, Expired, Invalid, Not recognized, Not yet
valid), the **Licensee**, the **Edition**, the **Limits** (users, servers and GPUs) and the expiry date
(or **Perpetual**).

To install it, paste the content of the `license.json` sent by Preteco into **Install a license** and
click **Install**. The license is a signed file validated offline.

!!! note
    In this version the license is reported but **does not restrict usage**.

### Telemetry

**Settings → Usage telemetry** controls whether Perceptron sends usage statistics. It is **disabled** by
default and anonymous: version, operating system, aggregated hardware and how often each feature is
used. **Never** your data, column names, paths or model metrics.

- **See exactly what would be sent** shows the content before you enable it.
- **Enable telemetry** / **Disable telemetry**.
- If no destination is configured, nothing is sent even when enabled.
