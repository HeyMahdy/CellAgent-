CellAgent

## Emergency stop

If an agent run or notebook kernel does not respond to Ctrl+C, run this from a
second terminal:

```bash
curl -X POST http://127.0.0.1:8000/api/stop-all
```

This terminates all active CellAgent runs, their notebook/SCSA subprocesses,
and the API server. Restart the API server before submitting another run.
