import uvicorn

if __name__ == "__main__":
    # Loopback only: the endpoints are unauthenticated and the pipeline posts from the same machine.
    uvicorn.run(
        "drishti.api.dashboard_service:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
    )
