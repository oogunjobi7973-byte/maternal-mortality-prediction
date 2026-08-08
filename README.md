# Early Maternal Mortality Prediction System

An explainable machine learning clinical decision support prototype for maternal mortality risk prediction in rural Nigerian healthcare settings.

## Technology stack

- Streamlit frontend
- FastAPI backend
- Explainable Boosting Machine (EBM)
- SHAP explainability
- PostgreSQL database
- JWT-based basic authentication
- Render deployment

## Database

The application uses PostgreSQL for persistent storage. On startup, SQLAlchemy creates the required tables if they do not already exist:

- `users`
- `patients`
- `predictions`
- `shap_explanations`

Prediction requests are associated with the authenticated healthcare worker. Patient assessment records, prediction results, and SHAP feature contributions are persisted in the database.

## Authentication

The prototype provides basic healthcare-worker account registration and login. Passwords are stored as secure password hashes, while authenticated API requests use short-lived JWT access tokens.

## Environment variables

Backend:

- `DATABASE_URL` - Render PostgreSQL connection string
- `JWT_SECRET` - long random secret used to sign access tokens

Frontend:

- `API_BASE_URL` - public URL of the deployed FastAPI backend

See `.env.example` for the expected format.
