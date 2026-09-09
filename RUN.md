# How to Run NEXUS LifeOS (Backend + Frontend)

Follow these steps to get the full stack running, including the **X-ray 3D** feature.

---

## 1. Backend (API + X-ray 3D reconstruction)

The backend serves the REST API at **http://localhost:8000** and provides the `/v1/nexus/xray/reconstruct-3d` endpoint for true 3D X-ray reconstruction.

### Option A — Windows (PowerShell or CMD)

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or use the helper script (installs deps if needed, then starts the server):

```cmd
cd backend
run.bat
```

### Option B — Linux / macOS

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
cd backend
chmod +x run.sh
./run.sh
```

### Verify backend

- Open **http://localhost:8000/health** — you should see JSON with `"status": "healthy"`.
- API base used by the frontend: **http://localhost:8000/v1/nexus**.

**Note:** The first time you use **True 3D** in the X-ray viewer with AI depth enabled, the backend may download the MiDaS model via `torch.hub`; subsequent requests will be faster.

---

## 2. Frontend (Angular)

The frontend expects the backend at **http://localhost:8000/v1/nexus** (see `frontend/src/environments/environment.ts`).

### Install and run

```bash
cd frontend
npm install
npm start
```

Or:

```bash
cd frontend
npm install
npx ng serve
```

- App is served at **http://localhost:4200** (or the URL shown in the terminal).
- Use the app to open **Clinical Intake** or the **3D X-ray Viewer**, upload an X-ray, then switch to **“True 3D”** to get the final 3D reconstruction (backend will be called automatically).

---

## 3. Getting the “final result” (X-ray 3D)

1. **Start backend** (step 1) and leave it running.
2. **Start frontend** (step 2) and open the app in the browser.
3. Go to **3D X-ray Viewer** (or Clinical Intake → Open 3D X-ray Viewer).
4. **Upload an X-ray** (e.g. JPG/PNG).
5. In the right panel, under **View Mode**, click **“True 3D”**.
6. Wait for **“Generating 3D model…”** (one-time per image); the backend runs depth estimation and mesh generation.
7. **Drag** to rotate the 3D mesh, **scroll** to zoom — that’s the final true 3D result.

---

## 4. Run both from project root (optional)

**Windows (two terminals):**

- Terminal 1: `cd backend && run.bat`
- Terminal 2: `cd frontend && npm install && npm start`

**Linux/macOS:**

- Terminal 1: `cd backend && ./run.sh`
- Terminal 2: `cd frontend && npm install && npm start`

Then open **http://localhost:4200** and use the X-ray 3D viewer as above.
