from openenv.core.env_server import create_fastapi_app
from server.environment import ResumeScreeningEnvironment
from models import ScreeningAction, ScreeningObservation
import uvicorn

app = create_fastapi_app(ResumeScreeningEnvironment, ScreeningAction, ScreeningObservation)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
