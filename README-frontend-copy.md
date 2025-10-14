# Using the UI from "1 - Copy"

You said the UI in "1 - Copy" looks better. This backend can now serve that UI from a drop-in folder named `frontend_copy/`.

How to use:

1) In this workspace, create a folder `frontend_copy/` next to `frontend/`.
2) Copy the UI files from your "1 - Copy" project into `frontend_copy/` (e.g., index.html, script.js, styles.css, and any assets).
3) Start the backend as usual. When `frontend_copy/` exists, it will be served at:
   - http://localhost:8000/  (index)
   - http://localhost:8000/frontend/* (assets)
4) The API endpoints remain identical:
   - POST /extract (multipart/form-data with field name `file` containing the PDF)
   - GET  /health
   - GET  /download/{filename}

Notes:
- The server rewrites relative `href` and `src` paths in `index.html` to `/frontend/...` so you don’t need to change the HTML.
- `window.API_BASE` is set to empty by default (same-origin). If your copied UI sets it earlier, that value is kept.
- Outputs are unchanged. CSVs still go to `outputs/` and the response schema stays the same.

If you need me to import files from your "1 - Copy" folder right now, share the path and I’ll add them here.